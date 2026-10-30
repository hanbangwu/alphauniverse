<script lang="ts">
  import PredictionsSwitch from '$lib/components/common/predictions-switch.svelte'
  import GalaxyName from '$lib/components/galaxy-name.svelte'
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
    onOpenAutoFocus={(event) => {
      event.preventDefault()
      content?.focus()
    }}
    class="z-100 flex max-h-11/12 w-full flex-col sm:max-w-[90vw]"
  >
    <Dialog.Header class="flex-row items-center gap-6">
      <Dialog.Title>Search from <GalaxyName {galaxy} /></Dialog.Title>
      <PredictionsSwitch />
    </Dialog.Header>

    <SearchForm />

    <div
      bind:this={content}
      tabindex="-1"
      class="flex min-h-0 flex-col gap-6 overflow-y-auto outline-none"
    >
      <SimilaritySelf />

      {#if similarity.searched}
        <Separator />
        <MatchList />
      {/if}
    </div>
  </Dialog.Content>
</Dialog.Root>
