<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import BookOpenIcon from '@lucide/svelte/icons/book-open'
  import GitBranchIcon from '@lucide/svelte/icons/git-branch'
</script>

<footer
  class="flex shrink-0 flex-col items-start gap-2 p-4 font-mono text-xs text-muted-foreground"
>
  <Button
    href="https://github.com/hanbangwu/alphauniverse"
    target="_blank"
    variant="outline"
    size="xs"
  >
    <GitBranchIcon data-icon="inline-start" />
    GitHub
  </Button>
  <Button href="/docs" target="_blank" variant="outline" size="xs">
    <BookOpenIcon data-icon="inline-start" />
    Read the docs
  </Button>
</footer>
