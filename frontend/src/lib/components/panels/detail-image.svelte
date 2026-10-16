<script lang="ts">
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()
</script>

<PanelFrame>
  <GalaxyThumb {galaxy} />
</PanelFrame>
