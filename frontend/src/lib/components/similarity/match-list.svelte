<script lang="ts">
  import { Spinner } from '$lib/components/ui/spinner'
  import { getMask } from '$lib/state/app.svelte'
  import MaskControls from './mask-controls.svelte'
  import MatchRow from './match-row.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
  const display = getMask()
</script>

<span class="text-sm font-medium">Whole dataset</span>
<p class="text-xs text-muted-foreground">
  Image matches are compared {similarity.anywhere('ls_image')
    ? 'anywhere in the image'
    : 'at the same place in the image'}; spectrum matches {similarity.anywhere('desi_spectrum')
    ? 'at any wavelength'
    : 'at the same observed wavelength'}.
</p>

{#if similarity.error}
  <p role="alert" class="text-sm text-destructive">{similarity.error}</p>
{:else if !similarity.imageMaps}
  <div class="grid h-24 place-content-center">
    <Spinner class="size-6" />
  </div>
{:else}
  {#if display.on.value && similarity.imageDomain}
    <div class="flex flex-col gap-1 md:max-w-xs">
      <span class="text-xs text-muted-foreground">Match mask threshold</span>
      <MaskControls domain={similarity.imageDomain} control={display.matches} />
    </div>
  {/if}
  <div class={['flex flex-col transition-opacity', similarity.stale && 'opacity-50']}>
    {#each similarity.matches as { galaxy, index } (galaxy)}
      <MatchRow {galaxy} {index} />
    {/each}
  </div>
{/if}
