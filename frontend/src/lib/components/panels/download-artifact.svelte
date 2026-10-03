<script lang="ts">
  import { downloadUrl } from '$lib/client'
  import { Button } from '$lib/components/ui/button'
</script>

<div class="flex flex-col gap-2">
  <div class="text-xs tracking-wider text-muted-foreground uppercase">Downloads</div>
  <Button href={downloadUrl('encoded')} variant="ghost" size="sm"
    >Download encoded embeddings</Button
  >
  <Button href={downloadUrl('codebook')} variant="ghost" size="sm"
    >Download codebook embeddings</Button
  >
  <Button href={downloadUrl('tokens')} variant="ghost" size="sm">Download tokens</Button>
</div>
