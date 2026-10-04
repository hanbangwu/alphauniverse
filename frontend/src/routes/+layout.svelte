<script lang="ts">
  import { browser } from '$app/environment'
  import favicon from '$lib/assets/favicon.svg'
  import AppRoot from '$lib/components/app-root.svelte'
  import { Spinner } from '$lib/components/ui/spinner'
  import { metaQuery } from '$lib/data/queries'
  import './layout.css'
  import { QueryClient, QueryClientProvider, createQuery } from '@tanstack/svelte-query'
  import { ModeWatcher } from 'mode-watcher'

  let { children } = $props()

  const queries = new QueryClient({ defaultOptions: { queries: { enabled: browser } } })
  const meta = createQuery(
    () => metaQuery,
    () => queries
  )
</script>

<svelte:head>
  <title>alphaUniverse</title>
  <link rel="icon" href={favicon} />
  <meta name="description" content="The embedding-based virtual astronomical observatory." />
</svelte:head>

<ModeWatcher />

<QueryClientProvider client={queries}>
  <div class="h-dvh overflow-hidden">
    {#if meta.data}
      <AppRoot meta={meta.data}>
        {@render children()}
      </AppRoot>
    {:else if meta.isError}
      <p role="alert" class="grid h-full place-content-center text-sm text-muted-foreground">
        Could not reach the server. Reload the page to try again.
      </p>
    {:else}
      <div class="grid h-full place-content-center">
        <Spinner class="size-8" />
      </div>
    {/if}
  </div>
</QueryClientProvider>
