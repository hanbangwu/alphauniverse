<script lang="ts">
  import favicon from '$lib/assets/favicon.svg'
  import { AppState, setApp } from '$lib/state/app.svelte'
  import './layout.css'
  import { QueryClient } from '@tanstack/query-core'
  import { QueryClientProvider } from '@tanstack/svelte-query'
  import { ModeWatcher } from 'mode-watcher'
  import { untrack } from 'svelte'

  let { data, children } = $props()

  const queries = new QueryClient()

  setApp(untrack(() => new AppState(data.meta))).start()
</script>

<svelte:head>
  <title>alphaUniverse</title>
  <link rel="icon" href={favicon} />
  <meta name="description" content="The embedding-based virtual astronomical observatory." />
</svelte:head>

<ModeWatcher />

<QueryClientProvider client={queries}>
  <div class="h-dvh overflow-hidden">
    {@render children()}
  </div>
</QueryClientProvider>
