<script lang="ts">
  import { browser } from '$app/environment'
  import markDark from '$lib/assets/alphauniverse-mark-dark.svg'
  import markLight from '$lib/assets/alphauniverse-mark-light.svg'
  import { client } from '$lib/client'
  import AppRoot from '$lib/components/app-root.svelte'
  import { Button } from '$lib/components/ui/button'
  import { Spinner } from '$lib/components/ui/spinner'
  import { metaQuery, pointsQuery, propertiesQuery } from '$lib/data/queries'
  import { POINT_SETS, type PointSet } from '$lib/labels'
  import { MosaicState } from '$lib/state/mosaic.svelte'
  import XIcon from '@lucide/svelte/icons/x'
  import { QueryClient, QueryClientProvider, createQuery } from '@tanstack/svelte-query'
  import { PersistedState } from 'runed'
  import { onDestroy, onMount } from 'svelte'

  let { children } = $props()
  const showDisclaimer = new PersistedState('show-disclaimer', true)
  const disclaimer =
    "This is a research prototype. We don't guarantee accuracy nor reliability (for now)."

  const queries = new QueryClient({ defaultOptions: { queries: { enabled: browser } } })
  const meta = createQuery(
    () => metaQuery,
    () => queries
  )
  const mosaic = new MosaicState()

  createQuery(
    () => pointsQuery(mosaic, Object.keys(POINT_SETS)[0] as PointSet),
    () => queries
  )
  createQuery(
    () => propertiesQuery(mosaic),
    () => queries
  )
  onDestroy(() => mosaic.destroy())

  function requirePassword(response: Response): Response {
    if (response.status === 401) location.assign('/login')
    return response
  }

  onMount(() => {
    client.interceptors.response.use(requirePassword)
    return () => client.interceptors.response.eject(requirePassword)
  })
</script>

<QueryClientProvider client={queries}>
  <div class="flex h-dvh flex-col overflow-hidden">
    {#if showDisclaimer.current && !meta.isPending}
      <div class="flex items-center border-b bg-muted px-4 py-1 text-xs text-muted-foreground">
        <p class="flex-1 text-center">{disclaimer}</p>
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
        <div class="flex h-full flex-col items-center p-4">
          <div class="flex flex-1 flex-col items-center justify-center gap-6">
            <h1 class="flex items-center gap-3 text-2xl tracking-wide">
              <img src={markLight} alt="" class="size-12 dark:hidden" />
              <img src={markDark} alt="" class="hidden size-12 dark:block" />
              alphaUniverse
            </h1>
            <Spinner class="size-8" />
          </div>
          <p class="text-center text-xs text-muted-foreground">{disclaimer}</p>
        </div>
      {/if}
    </div>
  </div>
</QueryClientProvider>
