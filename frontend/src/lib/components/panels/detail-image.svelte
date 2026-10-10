<script lang="ts">
  import ImageTokenFrame from '$lib/components/common/image-token-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()
</script>

<ImageTokenFrame>
  <GalaxyThumb {galaxy} />
</ImageTokenFrame>
