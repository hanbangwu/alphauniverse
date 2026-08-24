/** @type {import("prettier").Config} */
const config = {
  semi: false,
  singleQuote: true,
  trailingComma: 'none',
  printWidth: 100,
  plugins: [
    'prettier-plugin-svelte',
    'prettier-plugin-tailwindcss',
    '@trivago/prettier-plugin-sort-imports'
  ],
  overrides: [{ files: '*.svelte', options: { parser: 'svelte' } }],
  tailwindStylesheet: './src/routes/layout.css'
}

export default config
