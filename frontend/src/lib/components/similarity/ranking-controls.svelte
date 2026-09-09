<script lang="ts">
  import FieldSlider from '$lib/components/common/field-slider.svelte'
  import { getSearch } from '$lib/state/app.svelte'

  const search = getSearch()
</script>

<div class="flex flex-wrap items-center gap-x-6 gap-y-3">
  <span class="text-sm font-medium">Whole dataset</span>

  <label
    class="flex items-center gap-2 text-sm font-medium"
    title="How many galaxies the search returns."
  >
    Show top
    <FieldSlider field={search.matches} />
    <span class="w-7 text-right font-mono text-xs tabular-nums">{search.matches.value}</span>
  </label>
</div>
