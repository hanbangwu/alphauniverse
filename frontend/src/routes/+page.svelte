<script lang="ts">
  import LeftPanel from '$lib/components/panels/left-panel.svelte'
  import ProjectionView from '$lib/components/panels/projection-view.svelte'
  import RightPanel from '$lib/components/panels/right-panel.svelte'
  import * as Resizable from '$lib/components/ui/resizable'
</script>

<Resizable.PaneGroup direction="horizontal">
  <Resizable.Pane defaultSize={20} minSize={16}>
    <LeftPanel />
  </Resizable.Pane>

  <Resizable.Handle withHandle />

  <Resizable.Pane defaultSize={60} minSize={32}>
    <ProjectionView />
  </Resizable.Pane>

  <Resizable.Handle withHandle />

  <Resizable.Pane defaultSize={20} minSize={16}>
    <RightPanel />
  </Resizable.Pane>
</Resizable.PaneGroup>
