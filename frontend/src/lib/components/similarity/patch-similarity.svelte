<script lang="ts">
  import * as Dialog from '$lib/components/ui/dialog'
  import { Separator } from '$lib/components/ui/separator/index.js'
  import { getApp } from '$lib/state/app.svelte'
  import MaskControls from './mask-controls.svelte'
  import MatchList from './match-list.svelte'
  import SimilarityControls from './similarity-controls.svelte'
  import SimilaritySelf from './similarity-self.svelte'
  import { Similarity, setSimilarity } from './similarity.svelte'
  import { untrack } from 'svelte'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const { view, mask: display } = getApp()

  const similarity = setSimilarity(new Similarity(untrack(() => galaxy)))
</script>

<Dialog.Root open={view.explorer.value} onOpenChange={(open) => (view.explorer.value = open)}>
  <Dialog.Content class="z-100 max-h-11/12 w-full overflow-y-auto sm:max-w-6xl">
    <Dialog.Header>
      <Dialog.Title>Search</Dialog.Title>
    </Dialog.Header>

    <div class="flex flex-wrap items-center gap-4">
      <SimilarityControls />
      {#if display.on.value && similarity.imageDomain}
        <MaskControls domain={similarity.imageDomain} />
      {/if}
    </div>

    <SimilaritySelf />

    {#if view.querying}
      <Separator />
      <MatchList />
    {/if}
  </Dialog.Content>
</Dialog.Root>
