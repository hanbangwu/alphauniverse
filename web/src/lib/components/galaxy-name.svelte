<script lang="ts">
  import type { GalaxyId } from '$lib/client'

  interface Props {
    galaxy: GalaxyId
  }

  let { galaxy }: Props = $props()
</script>

{galaxy}
