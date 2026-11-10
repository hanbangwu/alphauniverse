<script lang="ts">
  import Dock from '$lib/components/dock.svelte'
  import GalaxyDrawer from '$lib/components/panels/galaxy-drawer.svelte'
  import ProjectionView from '$lib/components/panels/projection-view.svelte'
  import SkyView from '$lib/components/panels/sky-view.svelte'
  import TextSearch from '$lib/components/panels/text-search.svelte'
  import UmapDrawer from '$lib/components/panels/umap-drawer.svelte'
  import * as Tabs from '$lib/components/ui/tabs'
  import { getView } from '$lib/state/app.svelte'

  const view = getView()
</script>

<Tabs.Root bind:value={view.tab.value} class="relative h-full gap-0">
  <Dock />
  <Tabs.Content value="sky" class="absolute inset-0">
    <SkyView />
  </Tabs.Content>
  <Tabs.Content value="umap" class="absolute inset-0">
    <ProjectionView />
  </Tabs.Content>
  <Tabs.Content value="text" class="h-full overflow-y-auto pt-16">
    <TextSearch />
  </Tabs.Content>
  <UmapDrawer />
  <GalaxyDrawer />
</Tabs.Root>
