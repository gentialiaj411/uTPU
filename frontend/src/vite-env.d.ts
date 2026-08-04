/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DEFAULT_MODE?: "public" | "dev";
  readonly VITE_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
