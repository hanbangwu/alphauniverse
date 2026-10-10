<script lang="ts">
  import { Spinner } from '$lib/components/ui/spinner'
  import MatchRow from './match-row.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
</script>

<span class="text-sm font-medium">Whole dataset</span>
<p class="text-xs text-muted-foreground">
  Matches are compared at the same place in the image and the same observed wavelength.
</p>

{#if similarity.error}
  <p role="alert" class="text-sm text-destructive">{similarity.error}</p>
{:else if !similarity.imageMaps}
  <div class="grid h-24 place-content-center">
    <Spinner class="size-6" />
  </div>
{:else}
  <div class={['flex flex-col transition-opacity', similarity.stale && 'opacity-50']}>
    {#each similarity.matches as { galaxy, index } (galaxy)}
      <MatchRow {galaxy} {index} />
    {/each}
  </div>
{/if}
