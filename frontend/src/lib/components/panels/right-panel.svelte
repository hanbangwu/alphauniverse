<script lang="ts">
  import CollapsibleGroup from '$lib/components/common/collapsible-group.svelte'
  import Observed from '$lib/components/common/observed.svelte'
  import PredictionsSwitch from '$lib/components/common/predictions-switch.svelte'
  import SimilarityDialog from '$lib/components/similarity/similarity-dialog.svelte'
  import { Button } from '$lib/components/ui/button'
  import { IMAGE_SURVEYS, SPECTRUM_SURVEYS } from '$lib/data/modes'
  import { SURVEYS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import CoverageTable from './coverage-table.svelte'
  import DetailImage from './detail-image.svelte'
  import DetailSpectrum from './detail-spectrum.svelte'

  const { view } = getApp()
</script>

<div class="flex h-full flex-col bg-card">
  <h2 class="shrink-0 p-4 pb-2 text-xs font-medium tracking-wider text-muted-foreground uppercase">
    Details
  </h2>

  <div class="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 pb-4">
    {#if view.galaxy.value === null}
      <p class="text-sm leading-relaxed text-muted-foreground">No galaxy selected.</p>
    {:else}
      {@const galaxy = view.galaxy.value}
      <span class="text-sm font-medium">#{galaxy}</span>

      <PredictionsSwitch />

      <Button variant="secondary" size="sm" onclick={() => (view.explorer.value = true)}>
        Search
      </Button>

      <CollapsibleGroup title="Images" bind:open={view.open.current.images}>
        {#each IMAGE_SURVEYS as survey (survey)}
          <Observed {galaxy} {survey} kind="image">
            <div class="flex flex-col gap-1">
              <span class="text-xs text-muted-foreground">{SURVEYS[survey].label}</span>
              <DetailImage {galaxy} {survey} />
            </div>
          </Observed>
        {/each}
      </CollapsibleGroup>

      <CollapsibleGroup title="Spectra" bind:open={view.open.current.spectra}>
        {#each SPECTRUM_SURVEYS as survey (survey)}
          <Observed {galaxy} {survey} kind="spectrum">
            {#snippet children(matched)}
              <div class="flex flex-col gap-1">
                <span class="text-xs text-muted-foreground">{SURVEYS[survey].label}</span>
                <DetailSpectrum {galaxy} {survey} {matched} />
              </div>
            {/snippet}
          </Observed>
        {/each}
      </CollapsibleGroup>

      <CoverageTable {galaxy} />

      {#key galaxy}
        <SimilarityDialog {galaxy} />
      {/key}
    {/if}
  </div>
</div>
