<script lang="ts">
  import { Spinner } from '$lib/components/ui/spinner'
  import MatchRow from './match-row.svelte'
  import RankingControls from './ranking-controls.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
</script>

<RankingControls />

{#if !similarity.blended}
  <div class="grid h-24 place-content-center">
    <Spinner class="size-6" />
  </div>
{:else}
  <div class="flex flex-col">
    {#each similarity.matches as { galaxy, index } (galaxy)}
      <MatchRow {galaxy} {index} />
    {/each}
  </div>
{/if}
