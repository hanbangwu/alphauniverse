<script lang="ts">
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()
</script>

<PatchFrame>
  <GalaxyThumb {galaxy} />
</PatchFrame>
