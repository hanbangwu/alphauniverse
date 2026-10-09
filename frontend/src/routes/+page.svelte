<script lang="ts">
  import LeftPanel from '$lib/components/panels/left-panel.svelte'
  import ProjectionView from '$lib/components/panels/projection-view.svelte'
  import RightPanel from '$lib/components/panels/right-panel.svelte'
  import TextSearch from '$lib/components/panels/text-search.svelte'
  import * as Resizable from '$lib/components/ui/resizable'
  import * as Tabs from '$lib/components/ui/tabs'
</script>

<Resizable.PaneGroup direction="horizontal">
  <Resizable.Pane defaultSize={20} minSize={16}>
    <LeftPanel />
  </Resizable.Pane>

  <Resizable.Handle withHandle />

  <Resizable.Pane defaultSize={60} minSize={32}>
    <Tabs.Root value="umap" class="relative h-full gap-0">
      <Tabs.List
        class="absolute top-3 left-1/2 z-10 -translate-x-1/2 border bg-background/50 shadow-lg backdrop-blur-xl backdrop-saturate-150"
      >
        <Tabs.Trigger value="umap">Parametric UMAP</Tabs.Trigger>
        <Tabs.Trigger value="text">Text Search</Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content value="umap" class="min-h-0">
        <ProjectionView />
      </Tabs.Content>
      <Tabs.Content value="text" class="min-h-0 overflow-y-auto pt-12">
        <TextSearch />
      </Tabs.Content>
    </Tabs.Root>
  </Resizable.Pane>

  <Resizable.Handle withHandle />

  <Resizable.Pane defaultSize={20} minSize={16}>
    <RightPanel />
  </Resizable.Pane>
</Resizable.PaneGroup>
