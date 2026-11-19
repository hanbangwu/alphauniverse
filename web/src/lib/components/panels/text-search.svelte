<script lang="ts">
  import type { GetTextSearchDownloadData } from '$lib/api'
  import { client } from '$lib/client'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyName from '$lib/components/galaxy-name.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { Button } from '$lib/components/ui/button'
  import { Input } from '$lib/components/ui/input'
  import { Spinner } from '$lib/components/ui/spinner'
  import { textSearchQuery } from '$lib/data/queries'
  import type { ModeName } from '$lib/data/similarity'
  import { must } from '$lib/invariant'
  import { MODES } from '$lib/labels'
  import { getView } from '$lib/state/app.svelte'
  import { Field } from '$lib/state/field.svelte'
  import { DEFAULTS, stored } from '$lib/state/params.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  const view = getView()
  const submitted = new Field(DEFAULTS.text, stored('text'))
  let draft = $state(submitted.value ?? '')

  const results = createQuery(() => textSearchQuery(submitted.value))
  const download = $derived(
    client.buildUrl<GetTextSearchDownloadData>({
      url: '/search/text/download',
      query: { text: must(submitted.value, 'the searched text') }
    })
  )

  function leading(contributions: [ModeName, number][]): string {
    const lead = contributions.reduce((total, [, value]) => total + value, 0)
    return contributions
      .filter(([, value]) => lead > 0 && value >= 0.1 * lead)
      .toSorted(([, first], [, second]) => second - first)
      .map(([name, value]) => `${MODES[name]} ${Math.round((100 * value) / lead)}%`)
      .join(', ')
  }
</script>

<div class="mx-auto flex max-w-2xl flex-col gap-4 p-4">
  <form
    class="flex items-center gap-2"
    onsubmit={(event) => {
      event.preventDefault()
      submitted.value = draft.trim()
    }}
  >
    <Input bind:value={draft} maxlength={500} placeholder="Describe a galaxy" />
    <Button
      type="submit"
      size="sm"
      class="disabled:pointer-events-auto disabled:cursor-not-allowed"
      disabled={draft.trim() === ''}>Search</Button
    >
    {#if results.data && !results.isError}
      <Button
        variant="secondary"
        size="sm"
        disabled={results.isPlaceholderData}
        href={download}
        download="alphauniverse-text-search.ecsv">Export as ECSV</Button
      >
    {/if}
  </form>

  {#if results.isError}
    <p role="alert" class="text-sm text-destructive">{results.error.message}</p>
  {:else if results.data}
    {@const { galaxies, scores, contributions } = results.data}
    {@const named = Object.entries(contributions) as [ModeName, number[]][]}
    <ol
      class={['flex flex-col gap-1 transition-opacity', results.isPlaceholderData && 'opacity-50']}
    >
      {#each galaxies as galaxy, index (galaxy)}
        <li>
          <button
            type="button"
            onclick={() => view.select(galaxy)}
            aria-pressed={view.galaxy.value === galaxy}
            class="flex w-full items-center gap-4 rounded-lg p-2 text-left hover:bg-muted aria-pressed:bg-muted"
          >
            <PanelFrame class="size-20 shrink-0">
              <GalaxyThumb {galaxy} />
            </PanelFrame>
            <div class="flex flex-col gap-1">
              <span class="text-sm font-medium"><GalaxyName {galaxy} /></span>
              <span class="text-xs text-muted-foreground tabular-nums">
                cosine {scores[index].toFixed(3)} &bull; {leading(
                  named.map(([name, values]) => [name, values[index]])
                )}
              </span>
            </div>
          </button>
        </li>
      {/each}
    </ol>
  {:else if results.isFetching}
    <div class="grid h-16 place-content-center">
      <Spinner class="size-5" />
    </div>
  {/if}
</div>
