<script lang="ts">
  import favicon from '$lib/assets/favicon.svg'
  import './layout.css'
  import { ModeWatcher } from 'mode-watcher'

  let { children } = $props()
</script>

<svelte:head>
  <title>alphaUniverse</title>
  <link rel="icon" href="/favicon.ico" sizes="32x32" />
  <link rel="icon" href={favicon} type="image/svg+xml" />
  <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
  <meta name="description" content="The embedding-based virtual astronomical observatory." />
</svelte:head>

<ModeWatcher />

{@render children()}
