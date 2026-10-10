<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import { Input } from '$lib/components/ui/input'
  import { Spinner } from '$lib/components/ui/spinner'
  import { textSearchQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import TextSearchResult from './text-search-result.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  let draft = $state('')
  let submitted = $state<string | null>(null)

  const results = createQuery(() => textSearchQuery(submitted))
</script>

<div class="mx-auto flex max-w-2xl flex-col gap-4 p-4">
  <form
    class="flex items-center gap-2"
    onsubmit={(event) => {
      event.preventDefault()
      submitted = draft.trim()
    }}
  >
    <Input bind:value={draft} maxlength={500} placeholder="Describe a galaxy" />
    <Button
      type="submit"
      size="sm"
      class="disabled:pointer-events-auto disabled:cursor-not-allowed"
      disabled={draft.trim() === ''}>Search</Button
    >
  </form>

  {#if results.isError}
    <p role="alert" class="text-sm text-destructive">{errorMessage(results.error)}</p>
  {:else if results.data}
    {@const { galaxies, scores } = results.data}
    <ol
      class={['flex flex-col gap-1 transition-opacity', results.isPlaceholderData && 'opacity-50']}
    >
      {#each galaxies as galaxy, index (galaxy)}
        <li><TextSearchResult {galaxy} score={scores[index]} /></li>
      {/each}
    </ol>
  {:else if results.isFetching}
    <div class="grid h-16 place-content-center">
      <Spinner class="size-5" />
    </div>
  {/if}
</div>
