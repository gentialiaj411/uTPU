import type { AppMode } from "../lib/mode";
import { isDevMode } from "../lib/mode";

interface Props {
  sourceArtifact: string;
  mode: AppMode;
  githubRepo?: string;
  raw?: unknown;
  onInspectRaw?: () => void;
}

export function ArtifactLink({
  sourceArtifact,
  mode,
  githubRepo = "gentialiaj411/uTPU",
  onInspectRaw,
}: Props) {
  const githubUrl = `https://github.com/${githubRepo}/blob/main/${sourceArtifact}`;

  if (isDevMode(mode) && onInspectRaw) {
    return (
      <button type="button" className="artifact-link" onClick={onInspectRaw}>
        view source JSON
      </button>
    );
  }

  return (
    <a className="artifact-link" href={githubUrl} target="_blank" rel="noreferrer">
      view source JSON
    </a>
  );
}

export function RawArtifactPanel({ raw, path }: { raw: unknown; path: string }) {
  return (
    <pre className="raw-panel">
      <code>{path}{"\n"}{JSON.stringify(raw, null, 2)}</code>
    </pre>
  );
}
