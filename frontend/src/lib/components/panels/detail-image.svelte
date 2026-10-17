<script lang="ts">
  import type { ImageSurvey } from '$lib/client'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { getView } from '$lib/state/app.svelte'

  interface Props {
    galaxy: number
    survey: ImageSurvey
  }

  let { galaxy, survey }: Props = $props()

  const view = getView()
</script>

<PanelFrame>
  <GalaxyThumb {galaxy} {survey} predicted={view.predictions.value} />
</PanelFrame>
