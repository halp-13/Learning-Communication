<template>
    <div class="container-fluid h-100">
        <div class="row justify-content-center h-100">
            <div class="col-8 h-100 p-0">
                <div class="card h-100">
                    <div class="card-header" style="background-color: #4c4c4c;">
                        <h4 class="card-title text-center">Alice-bob</h4>
                    </div>
                    <div class="card-body d-flex p-0">
                        <!-- <canvas ref="canvas" id="canvas" width="200" height="100"></canvas>
                          -->

            <div class="d-flex align-items-center justify-content-center " style="flex:1;" ref="canvasBox">
                <canvas style="margin:0;padding:0; background-repeat: no-repeat; background-position:center; " ref="canvas" id="canvas" class="canvasView" ></canvas> 
            </div>
                    </div>
                </div>
            </div>
            <div class="col-4 h-100">
                <div class="card h-100" style="background-color: #4c4c4c;">
                    <div class="card-header">
                        <h4 class="card-title text-center">Alice-bob</h4>
                    </div>
                    <div class="card-body">
                        <div class="d-flex justify-content-center">
                            <button class="btn btn-primary">Add Node</button>
                        </div>
                        <div class="d-flex justify-content-center">
                            <button class="btn btn-primary">Add Edge</button>
                        </div>
                        <div class="d-flex justify-content-center">
                            <button class="btn btn-primary">Delete Node</button>
                        </div>
                        <div class="d-flex justify-content-center">
                            <button class="btn btn-primary">Delete Edge</button>
                        </div>
                    </div>

                    <div class="card-footer">
                        <h4 class="card-title
                        text-center">Alice-bob</h4>

                    </div>

                </div>
            <!-- </div> -->



        </div>
    </div>
    </div>

</template>
<script setup lang="ts">
import { ref, onMounted } from 'vue';
// import { createGraph } from '../utils/graph';
// import { Graph } from 'graphology';
// import { renderGraph } from '../utils/render';
// import { addNode, addEdge, deleteNode, deleteEdge } from '../utils/graph';

// get canvas element from template



const canvas = ref<HTMLCanvasElement | null>(null);
const ctx = ref<CanvasRenderingContext2D | null>(null);
const canvasBox = ref<HTMLElement | null>(null);

onMounted(() => {
    const canvasElement = canvas.value;
    if (canvasElement !== null) {
        console.log('canvasElement', canvasElement);
        const context = canvasElement.getContext('2d');
        if (context) {
            ctx.value = context;
            // draw cercle in middle of canvas

            // console.log('{canvasElement.width}', canvasElement.width);


         

        }
    }
    window.addEventListener('resize', () => {resizeCanvas();});
    resizeCanvas();
    console.log('mounted');
});

const resizeCanvas = () => {
    const canvasElement = canvas.value;
    const canvasBoxElement = canvasBox.value;
    if (canvasElement !== null && canvasBoxElement !== null) {
        canvasElement.width = canvasBoxElement.clientWidth;
        canvasElement.height = canvasBoxElement.clientHeight;
    }

    console.log('resizeCanvas');
    draw();



};
 
const draw = () => {
    const context = ctx.value;
    if (context) {
        context.clearRect(0, 0, context.canvas.width, context.canvas.height);
        context.beginPath();
        context.arc(context.canvas.width / 2, context.canvas.height / 2, 50, 0, 2 * Math.PI);
        context.stroke();
    }
    console.log('draw');
};


</script>


