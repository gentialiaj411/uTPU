import json
from pathlib import Path
import numpy as np

from firmware.host.numerics_v2.specdec_verify import verify_k
from tools.decode.harness import ROOT, load_runtime_260k
from tools.specdec.measure import draft_tokens, offline_passes
from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama


def _rt15():
    pay = np.load(ROOT / 'models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz', allow_pickle=True)['payload'][0]
    pay.pop('dynamic_base_scales', None)
    return RunQIntegerLlama(pay, ROOT / 'firmware/host/numerics_v2/luts', kv_bits=4)


def test_verify_k_lossless_and_rollback():
    rt = _rt15()
    prompt = [1, 9038, 2501, 263]
    for p, tok in enumerate(prompt[:-1]):
        rt.forward_argmax(tok, p, 4096)
    # At position 3, a deliberately wrong first draft exercises rollback.
    pos = len(prompt) - 1
    baseline = _rt15()
    for p, tok in enumerate(prompt[:-1]):
        baseline.forward_argmax(tok, p, 4096)
    expected = baseline.forward_argmax(prompt[-1], pos, 4096)
    before = (rt.key_code.copy(), rt.key_scale.copy(), rt.value_code.copy(), rt.value_scale.copy())
    got = verify_k(rt, pos, [expected, 0, 1, 2], token=prompt[-1])
    assert got['output'][0] == expected
    assert got['accepted'] == [expected]
    # State is after the accepted prefix, not after rejected speculative work.
    assert np.array_equal(rt.key_code, baseline.key_code)
    assert np.array_equal(rt.key_scale, baseline.key_scale)
    assert np.array_equal(rt.value_code, baseline.value_code)
    assert np.array_equal(rt.value_scale, baseline.value_scale)


def test_verify_k_matches_plain_greedy_for_multiple_k():
    prompts = [[1, 9038, 2501, 263], [1, 3118], [1, 9038, 2501, 263, 931]]
    for prompt in prompts:
        pos = len(prompt) - 1
        for k in (1, 2, 3, 4):
            rt = _rt15()
            for p, tok in enumerate(prompt[:-1]):
                rt.forward_argmax(tok, p, 4096)
            # Use plain greedy successors as the draft; this is the all-match path.
            tmp = _rt15()
            for p, tok in enumerate(prompt[:-1]):
                tmp.forward_argmax(tok, p, 4096)
            drafts, prev = [], prompt[-1]
            for i in range(k):
                nxt = tmp.forward_argmax(prev, pos + i, 4096)
                drafts.append(int(nxt)); prev = int(nxt)
            got = verify_k(rt, pos, drafts, token=prompt[-1])
            assert got['accepted'] == drafts
            assert got['output'] == drafts + [got['token']]


def test_saved_offline_acceptance_matches_verify_k_end_to_end():
    """The saved-sequence shortcut must agree with real verify_k passes."""
    artifact = ROOT / 'bench/results/specdec_sequences.json'
    assert artifact.exists(), 'run tools/specdec/measure.py first'
    saved = json.loads(artifact.read_text())
    assert saved['status'] == 'pass'

    # Three prompts are enough to exercise both match and rollback paths while
    # keeping this differential test practical on the host reference model.
    for row in saved['sequences'][:3]:
        prompt = list(map(int, row['prompt_tokens']))
        seq = list(map(int, row['sequence_tokens']))
        offline = offline_passes(prompt, seq, 'context4', 3, [], {})

        rt = _rt15()
        for pos, token in enumerate(prompt[:-1]):
            rt.forward_argmax(token, pos, 4096)
        pending = prompt[-1]
        position = len(prompt) - 1
        target_index = len(prompt)
        observed = []
        while target_index < len(seq):
            drafts = draft_tokens('context4', seq[:target_index],
                                  target_index - len(prompt), 3, [], {})
            got = verify_k(rt, position, drafts, token=pending)
            accepted = len(got['accepted'])
            emit = min(accepted + 1, len(seq) - target_index)
            assert got['output'] == seq[target_index:target_index + emit]
            observed.append(accepted)

            # verify_k rolls back rejected work and replays through the last
            # accepted prediction. Consume that last accepted draft so the
            # next pass starts with exactly the same state as greedy decoding.
            if accepted < len(drafts):
                if accepted:
                    rt.forward_argmax(got['accepted'][-1], position + accepted, 4096)
                else:
                    # With zero accepted drafts verify_k restores the snapshot
                    # before the pending context token; commit that token before
                    # starting the next pass at the returned fallback.
                    rt.forward_argmax(pending, position, 4096)
            target_index += emit
            position = len(rt.tokens)
            pending = seq[target_index - 1]

        assert observed == offline['accepted']

