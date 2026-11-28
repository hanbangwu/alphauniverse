<script lang="ts">
  import type { GalaxyId, ImageSurvey } from '$lib/client'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { getView } from '$lib/state/app.svelte'

  interface Props {
    galaxy: GalaxyId
    survey: ImageSurvey
  }

  let { galaxy, survey }: Props = $props()

  const view = getView()
</script>

<PanelFrame>
  <GalaxyThumb {galaxy} {survey} predicted={view.dataSource.value === 'prediction'} />
</PanelFrame>
