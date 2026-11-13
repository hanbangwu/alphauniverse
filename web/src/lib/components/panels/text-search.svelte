<script lang="ts">
  import type { GetTextSearchDownloadData } from '$lib/api'
  import { client } from '$lib/client'
  import { Button } from '$lib/components/ui/button'
  import { Input } from '$lib/components/ui/input'
  import { Spinner } from '$lib/components/ui/spinner'
  import { textSearchQuery } from '$lib/data/queries'
  import type { ModeName } from '$lib/data/similarity'
  import { must } from '$lib/invariant'
  import { Field } from '$lib/state/field.svelte'
  import { DEFAULTS, stored } from '$lib/state/params.svelte'
  import TextSearchResult from './text-search-result.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  const submitted = new Field(DEFAULTS.text, stored('text'))
  let draft = $state(submitted.value ?? '')

  const results = createQuery(() => textSearchQuery(submitted.value))
  const download = $derived(
    client.buildUrl<GetTextSearchDownloadData>({
      url: '/search/text/download',
      query: { text: must(submitted.value, 'the searched text') }
    })
  )
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
        download="alphauniverse-text-search.csv">Export as CSV</Button
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
          <TextSearchResult
            {galaxy}
            score={scores[index]}
            contributions={named.map(([name, values]) => [name, values[index]])}
          />
        </li>
      {/each}
    </ol>
  {:else if results.isFetching}
    <div class="grid h-16 place-content-center">
      <Spinner class="size-5" />
    </div>
  {/if}
</div>
