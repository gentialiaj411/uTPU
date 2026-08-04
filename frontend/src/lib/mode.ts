export type AppMode = "public" | "dev";

export function resolveMode(): AppMode {
  const params = new URLSearchParams(window.location.search);
  const urlMode = params.get("mode");
  if (urlMode === "dev") return "dev";
  const envMode = import.meta.env.VITE_DEFAULT_MODE;
  if (envMode === "dev") return "dev";
  return "public";
}

export function isDevMode(mode: AppMode): boolean {
  return mode === "dev";
}
