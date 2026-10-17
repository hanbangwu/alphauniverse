<script lang="ts">
  import { browser } from '$app/environment'
  import favicon from '$lib/assets/favicon.svg'
  import AppRoot from '$lib/components/app-root.svelte'
  import { Button } from '$lib/components/ui/button'
  import { Spinner } from '$lib/components/ui/spinner'
  import { metaQuery, pointsQuery } from '$lib/data/queries'
  import { POINT_SETS, type PointSet } from '$lib/labels'
  import { MosaicState } from '$lib/state/mosaic.svelte'
  import './layout.css'
  import XIcon from '@lucide/svelte/icons/x'
  import { QueryClient, QueryClientProvider, createQuery } from '@tanstack/svelte-query'
  import { ModeWatcher } from 'mode-watcher'
  import { PersistedState } from 'runed'
  import { onDestroy } from 'svelte'

  let { children } = $props()
  const showDisclaimer = new PersistedState('show-disclaimer', true)

  const queries = new QueryClient({ defaultOptions: { queries: { enabled: browser } } })
  const meta = createQuery(
    () => metaQuery,
    () => queries
  )
  const mosaic = new MosaicState()

  const shown = Object.keys(POINT_SETS)[0] as PointSet
  for (const projection of new Set<PointSet>(['mean', shown])) {
    createQuery(
      () => pointsQuery(mosaic, projection),
      () => queries
    )
  }
  onDestroy(() => mosaic.destroy())
</script>

<svelte:head>
  <title>alphaUniverse</title>
  <link rel="icon" href={favicon} />
  <meta name="description" content="The embedding-based virtual astronomical observatory." />
</svelte:head>

<ModeWatcher />

<QueryClientProvider client={queries}>
  <div class="flex h-dvh flex-col overflow-hidden">
    {#if showDisclaimer.current}
      <div class="flex items-center border-b bg-muted px-4 py-1 text-xs text-muted-foreground">
        <p class="flex-1 text-center">
          This is a research prototype. We don't guarantee accuracy nor reliability (for now).
        </p>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label="Close"
          onclick={() => (showDisclaimer.current = false)}
        >
          <XIcon />
        </Button>
      </div>
    {/if}
    <div class="min-h-0 flex-1">
      {#if meta.data}
        <AppRoot meta={meta.data} {mosaic}>
          {@render children()}
        </AppRoot>
      {:else if meta.isError}
        <p role="alert" class="grid h-full place-content-center text-sm text-muted-foreground">
          The server is unavailable. Reload the page to try again.
        </p>
      {:else}
        <div class="grid h-full place-content-center">
          <Spinner class="size-8" />
        </div>
      {/if}
    </div>
  </div>
</QueryClientProvider>
