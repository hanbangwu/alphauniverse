<script lang="ts">
  import { coverageQuery } from '$lib/data/queries'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const coverage = createQuery(() => coverageQuery(galaxy))
</script>

{coverage.data?.object_id ?? '…'}
