import js from '@eslint/js'
import prettier from 'eslint-config-prettier'
import svelte from 'eslint-plugin-svelte'
import { defineConfig, globalIgnores, includeIgnoreFile } from 'eslint/config'
import globals from 'globals'
import path from 'node:path'
import ts from 'typescript-eslint'

const gitignorePath = path.resolve(import.meta.dirname, '.gitignore')

export default defineConfig(
  includeIgnoreFile(gitignorePath),
  globalIgnores(['src/lib/components/ui/']),
  js.configs.recommended,
  ts.configs.recommended,
  svelte.configs.recommended,
  prettier,
  svelte.configs.prettier,
  {
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    rules: {
      'no-undef': 'off',
      'no-warning-comments': ['error', { terms: ['todo', 'fixme', 'hack', 'xxx'] }],
      'svelte/no-navigation-without-resolve': ['error', { ignoreReplaceState: true }]
    }
  },
  {
    files: ['**/*.svelte', '**/*.svelte.ts', '**/*.svelte.js'],
    languageOptions: {
      parserOptions: {
        projectService: true,
        extraFileExtensions: ['.svelte'],
        parser: ts.parser
      }
    }
  }
)
