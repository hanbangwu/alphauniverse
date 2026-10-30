<script lang="ts">
  import markDark from '$lib/assets/alphauniverse-mark-dark.svg'
  import markLight from '$lib/assets/alphauniverse-mark-light.svg'
  import Segmented from '$lib/components/common/segmented.svelte'
  import ThemeToggleButton from '$lib/components/theme-toggle-button.svelte'
  import { getView } from '$lib/state/app.svelte'
  import ColourControl from './colour-control.svelte'
  import DatasetFooter from './dataset-footer.svelte'
  import DownloadArtifact from './download-artifact.svelte'

  const view = getView()
</script>

<div class="glass flex h-full flex-col">
  <header class="flex shrink-0 items-start justify-between gap-2 p-4 pb-3">
    <h1 class="flex items-center gap-2 text-base tracking-wide">
      <img src={markLight} alt="" class="size-6 dark:hidden" />
      <img src={markDark} alt="" class="hidden size-6 dark:block" />
      alphaUniverse
    </h1>
    <ThemeToggleButton />
  </header>

  <div class="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4 pb-4">
    <div class="text-xs tracking-wider text-muted-foreground uppercase">Layout</div>
    <Segmented field={view.pointSet} />

    <ColourControl />

    <DownloadArtifact />
  </div>

  <DatasetFooter />
</div>
