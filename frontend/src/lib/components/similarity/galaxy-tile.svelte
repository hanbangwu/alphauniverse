<script lang="ts">
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()
</script>

<div class="flex flex-1 flex-col gap-2">
  <span class="text-sm font-medium">Row #{galaxy}</span>
  <PatchFrame>
    <GalaxyThumb {galaxy} />
  </PatchFrame>
</div>
