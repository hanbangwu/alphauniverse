<script lang="ts">
  import { coverageQuery } from '$lib/data/queries'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const coverage = createQuery(() => coverageQuery(galaxy))

  const EGGS: Record<number, string> = {
    42: '🌌', // meaning of the universe
    67: '🫲🤪🫱', // sIxTy SeVeNnnNnnnNNNNnnNnnnnnnnnNNnnNNnnn
    314: '🥧', // pi
    1031: '🎃', // halloween
    1225: '☃️' // christmas
  }
</script>

{coverage.data?.object_id ?? '…'}
{#if galaxy in EGGS}
  {EGGS[galaxy]}
{/if}
