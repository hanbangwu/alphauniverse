<script lang="ts">
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { getView } from '$lib/state/app.svelte'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const view = getView()
</script>

<PanelFrame>
  <GalaxyThumb {galaxy} predicted={view.predictions.value} />
</PanelFrame>
