<script lang="ts">
  import Segmented from '$lib/components/common/segmented.svelte'
  import ThemeToggleButton from '$lib/components/theme-toggle-button.svelte'
  import { getView } from '$lib/state/app.svelte'
  import DatasetFooter from './dataset-footer.svelte'
  import DownloadArtifact from './download-artifact.svelte'
  import MorphologyFilter from './morphology-filter.svelte'

  const view = getView()
</script>

<div class="flex h-full flex-col bg-card">
  <header class="flex shrink-0 items-start justify-between gap-2 p-4 pb-3">
    <h1 class="text-base tracking-wide">alphaUniverse</h1>
    <ThemeToggleButton />
  </header>

  <div class="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4 pb-4">
    <div class="text-xs tracking-wider text-muted-foreground uppercase">Embeddings</div>
    <Segmented field={view.pointSet} />

    <MorphologyFilter />

    <DownloadArtifact />
  </div>

  <DatasetFooter />
</div>
