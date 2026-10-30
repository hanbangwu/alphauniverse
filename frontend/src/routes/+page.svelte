<script lang="ts">
  import LeftPanel from '$lib/components/panels/left-panel.svelte'
  import ProjectionView from '$lib/components/panels/projection-view.svelte'
  import RightPanel from '$lib/components/panels/right-panel.svelte'
  import TextSearch from '$lib/components/panels/text-search.svelte'
  import * as Resizable from '$lib/components/ui/resizable'
  import * as Tabs from '$lib/components/ui/tabs'

  let rightPanel = $state(20)
</script>

<Tabs.Root value="umap" class="relative h-full gap-0" style="--right-panel: {rightPanel}%">
  <Tabs.Content value="umap" class="absolute inset-0">
    <ProjectionView />
  </Tabs.Content>

  <Resizable.PaneGroup direction="horizontal" class="pointer-events-none relative">
    <Resizable.Pane defaultSize={20} minSize={16} class="pointer-events-auto">
      <LeftPanel />
    </Resizable.Pane>

    <Resizable.Handle withHandle class="pointer-events-auto" />

    <Resizable.Pane defaultSize={60} minSize={32} class="relative">
      <Tabs.List class="pointer-events-auto absolute top-3 left-1/2 z-10 -translate-x-1/2">
        <Tabs.Trigger value="umap">Map</Tabs.Trigger>
        <Tabs.Trigger value="text">Text Search</Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content value="text" class="pointer-events-auto h-full overflow-y-auto pt-12">
        <TextSearch />
      </Tabs.Content>
    </Resizable.Pane>

    <Resizable.Handle withHandle class="pointer-events-auto" />

    <Resizable.Pane
      defaultSize={20}
      minSize={16}
      class="pointer-events-auto"
      onResize={(size) => (rightPanel = size)}
    >
      <RightPanel />
    </Resizable.Pane>
  </Resizable.PaneGroup>
</Tabs.Root>
