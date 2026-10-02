<script lang="ts">
  import * as Dialog from '$lib/components/ui/dialog'
  import { Separator } from '$lib/components/ui/separator/index.js'
  import { getApp } from '$lib/state/app.svelte'
  import MatchList from './match-list.svelte'
  import SearchForm from './search-form.svelte'
  import SimilaritySelf from './similarity-self.svelte'
  import { Similarity, setSimilarity } from './similarity.svelte'
  import { untrack } from 'svelte'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const { view } = getApp()

  const similarity = setSimilarity(new Similarity(untrack(() => galaxy)))

  let content = $state<HTMLElement | null>(null)
</script>

<Dialog.Root open={view.explorer.value} onOpenChange={(open) => (view.explorer.value = open)}>
  <Dialog.Content
    bind:ref={content}
    onOpenAutoFocus={(event) => {
      event.preventDefault()
      content?.focus()
    }}
    class="z-100 max-h-11/12 w-full overflow-y-auto sm:max-w-6xl"
  >
    <Dialog.Header>
      <Dialog.Title>Search</Dialog.Title>
    </Dialog.Header>

    <SimilaritySelf />

    <SearchForm />

    {#if similarity.searched}
      <Separator />
      <MatchList />
    {/if}
  </Dialog.Content>
</Dialog.Root>
