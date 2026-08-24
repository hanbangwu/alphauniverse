<script lang="ts">
  import { tokenColors } from '$lib/color'
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import PatchGrid from '$lib/components/patch/patch-grid.svelte'
  import { tokensQuery } from '$lib/data/queries'
  import { DETAIL_VIEWS } from '$lib/labels'
  import { getMeta, getView } from '$lib/state/app.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const meta = getMeta()
  const view = getView()

  const detail = $derived(view.detail.value)
  const showTokens = $derived(detail === 'tokens')

  const tokens = createQuery(() => tokensQuery(meta, galaxy))

  const values = $derived(showTokens ? (tokens.data ?? null) : null)

  const palette = $derived(values ? tokenColors(values) : null)

  const request = $derived(showTokens ? tokens : null)

  const failed = $derived.by(() => {
    if (!request?.isError) return null
    const error: unknown = request.error
    if (error instanceof Error) return error.message
    if (typeof error === 'object' && error !== null && 'detail' in error) {
      return String(error.detail)
    }
    return String(error)
  })
</script>

<PatchFrame busy={request?.isFetching ?? false}>
  <GalaxyThumb {galaxy} />
  {#if values && palette}
    <PatchGrid
      {values}
      grid={meta.grid}
      color={palette}
      title={(value) => `token ${value}`}
      label={`${DETAIL_VIEWS[detail].label} for galaxy ${galaxy}`}
      class="absolute inset-0"
    />
  {/if}
</PatchFrame>

{#if failed}
  <p class="text-xs leading-relaxed text-destructive">
    {DETAIL_VIEWS[detail].label} failed: {failed}
  </p>
{/if}
