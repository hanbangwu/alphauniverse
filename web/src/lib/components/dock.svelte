<script lang="ts">
  import markDark from '$lib/assets/alphauniverse-mark-dark.svg'
  import markLight from '$lib/assets/alphauniverse-mark-light.svg'
  import { Button } from '$lib/components/ui/button'
  import * as Tabs from '$lib/components/ui/tabs'
  import * as Tooltip from '$lib/components/ui/tooltip'
  import BookOpenIcon from '@lucide/svelte/icons/book-open'
  import GitBranchIcon from '@lucide/svelte/icons/git-branch'
  import MoonIcon from '@lucide/svelte/icons/moon'
  import SunIcon from '@lucide/svelte/icons/sun'
  import { toggleMode } from 'mode-watcher'
</script>

<Tooltip.Provider>
  <header
    class="glass absolute top-3 left-1/2 z-10 grid -translate-x-1/2 grid-cols-[1fr_auto_1fr] items-center gap-4 rounded-2xl p-1 pl-3"
  >
    <span>
      <img src={markLight} alt="alphaUniverse" class="size-6 dark:hidden" />
      <img src={markDark} alt="alphaUniverse" class="hidden size-6 dark:block" />
    </span>

    <Tabs.List variant="bare">
      <Tabs.Trigger value="sky">Sky</Tabs.Trigger>
      <Tabs.Trigger value="umap">UMAP</Tabs.Trigger>
      <Tabs.Trigger value="text">Text Search</Tabs.Trigger>
    </Tabs.List>

    <div class="flex justify-end">
      <Tooltip.Root>
        <Tooltip.Trigger>
          {#snippet child({ props })}
            <Button
              {...props}
              variant="ghost"
              size="icon"
              aria-label="Toggle theme"
              onclick={toggleMode}
            >
              <SunIcon class="dark:hidden" />
              <MoonIcon class="hidden dark:block" />
            </Button>
          {/snippet}
        </Tooltip.Trigger>
        <Tooltip.Content side="bottom" sideOffset={8}>Toggle theme</Tooltip.Content>
      </Tooltip.Root>

      <Tooltip.Root>
        <Tooltip.Trigger>
          {#snippet child({ props })}
            <Button
              {...props}
              href="https://github.com/hanbangwu/alphauniverse"
              target="_blank"
              variant="ghost"
              size="icon"
              aria-label="GitHub"
            >
              <GitBranchIcon />
            </Button>
          {/snippet}
        </Tooltip.Trigger>
        <Tooltip.Content side="bottom" sideOffset={8}>GitHub</Tooltip.Content>
      </Tooltip.Root>

      <Tooltip.Root>
        <Tooltip.Trigger>
          {#snippet child({ props })}
            <Button
              {...props}
              href="/docs"
              target="_blank"
              variant="ghost"
              size="icon"
              aria-label="Docs"
            >
              <BookOpenIcon />
            </Button>
          {/snippet}
        </Tooltip.Trigger>
        <Tooltip.Content side="bottom" sideOffset={8}>Docs</Tooltip.Content>
      </Tooltip.Root>
    </div>
  </header>
</Tooltip.Provider>
