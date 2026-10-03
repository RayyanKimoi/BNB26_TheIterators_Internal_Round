import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // The API base URL is configurable, so no proxy is needed. Set
    // VITE_API_BASE_URL when the backend is not on localhost:8000.
  },
});
