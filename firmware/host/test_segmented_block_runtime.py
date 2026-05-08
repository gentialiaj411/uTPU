import json
import numpy as np

from program_loader import ProgramLoader
from tiled_inference import TiledInferenceEngine, get_default_paths


def main():
    weights_dir, model_path, data_dir = get_default_paths()
    eng = TiledInferenceEngine(weights_dir, model_path, tiling_mode="array_block", verbose=False)
    loader = ProgramLoader(uart=None, verbose=False)
    a = eng.array_size

    img = np.load(data_dir + "\\mnist_14x14_test.npy")[0]
    x = eng.preprocess_image(img).astype(np.int8)
    w = eng.fc1_weight.astype(np.int8)

    seg = loader.build_fc_layer_block_program_segmented(
        weights_int4=w,
        activations_int4=x,
        out_features=w.shape[0],
        in_features=w.shape[1],
        array_size=a,
        apply_relu=False,
        apply_quant=True,
        max_words_per_segment=1024,
    )

    all_fit = all(words <= 1024 for words in seg["segment_words"])
    first_has_clear = seg["first_segment_has_acc_clear"]
    later_has_clear = seg["later_segments_have_acc_clear"]
    final_has_finalize = seg["final_segment_has_finalize_fetch"]
    earlier_has_finalize = seg["earlier_segments_have_finalize_fetch"]

    # Segmentation semantics check: total accumulate ops should equal in_blocks for FC1 (out_blocks=1).
    op_kinds = seg["segment_op_kinds"]
    total_acc_ops = sum(k == "accumulate" for seg_k in op_kinds for k in seg_k)
    total_finalize_ops = sum(k == "finalize_fetch" for seg_k in op_kinds for k in seg_k)

    in_blocks = (w.shape[1] + a - 1) // a
    expected_acc_ops = in_blocks

    # Int32 accumulation reference (unsegmented)
    ref_accum = eng.array_block_matmul_int32(w, x)[0]

    # Segmented-by-group accumulation simulation (accumulate ops are equivalent independent of segmentation).
    w_pad = np.zeros((a, in_blocks * a), dtype=np.int8)
    w_pad[:w.shape[0], :w.shape[1]] = w
    x_pad = np.zeros(in_blocks * a, dtype=np.int8)
    x_pad[:x.shape[0]] = x
    accum = np.zeros(a, dtype=np.int32)
    for ib in range(in_blocks):
        i0 = ib * a
        i1 = i0 + a
        accum += w_pad[:, i0:i1].astype(np.int32) @ x_pad[i0:i1].astype(np.int32)
    segmented_accum_match = bool(np.array_equal(accum[:w.shape[0]], ref_accum))

    results = {
        "segment_count": int(seg["segment_count"]),
        "segment_words": [int(x) for x in seg["segment_words"]],
        "all_segments_fit_1024": bool(all_fit),
        "first_segment_has_acc_clear": bool(first_has_clear),
        "later_segments_have_acc_clear": bool(later_has_clear),
        "final_segment_has_finalize_fetch": bool(final_has_finalize),
        "earlier_segments_have_finalize_fetch": bool(earlier_has_finalize),
        "total_accumulate_ops": int(total_acc_ops),
        "expected_accumulate_ops": int(expected_acc_ops),
        "total_finalize_ops": int(total_finalize_ops),
        "segmented_accum_match_unsegmented": bool(segmented_accum_match),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
