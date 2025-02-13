<script setup lang="ts">
import NodeCanvas from './components/node-canvas.vue';

</script>

<template>
  <div class="d-flex flex-column " style="flex:1; background:#32a852;  height:100vh" >
    <NuxtRouteAnnouncer />
    <!-- <NuxtWelcome />
      -->
    <NodeCanvas/>
  </div>
</template>
