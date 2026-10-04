import adapter from '@sveltejs/adapter-auto'
import { sveltekit } from '@sveltejs/kit/vite'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    rolldownOptions: {
      treeshake: {
        moduleSideEffects: [{ test: /\/node_modules\/embedding-atlas\//, sideEffects: false }]
      }
    }
  },
  plugins: [
    tailwindcss(),
    sveltekit({
      compilerOptions: {
        runes: ({ filename }) =>
          filename.split(/[/\\]/).includes('node_modules') ? undefined : true
      },

      adapter: adapter()
    })
  ]
})
