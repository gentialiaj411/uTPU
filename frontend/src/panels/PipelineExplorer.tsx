import { CompilerWalkthrough } from "../components/CompilerWalkthrough";

import type { Pipeline } from "../lib/data";

import type { AppMode } from "../lib/mode";

import { isDevMode } from "../lib/mode";

import { PipelineDagView } from "./PipelineDagView";



interface Props {

  pipelines: Pipeline[];

  mode: AppMode;

}



export function PipelineExplorer({ pipelines, mode }: Props) {

  const pipeline = pipelines[0];

  const dev = isDevMode(mode);



  if (!pipeline) {

    return (

      <section className="panel">

        <h2>Compiler Walkthrough</h2>

        <p>No pipeline data — run tools/build_frontend_data.py with PyTorch available.</p>

      </section>

    );

  }



  return (

    <section className="panel panel-compiler">

      <header className="panel-header">

        <h2>Compiler Walkthrough</h2>

        <p>

          <strong>{pipeline.name}</strong> · input {JSON.stringify(pipeline.example_input_shape)} ·

          array_size={pipeline.array_size}

        </p>

        <p className="coverage-note">{pipeline.coverage.asymmetry_note}</p>

      </header>



      <CompilerWalkthrough pipeline={pipeline} />



      <details className="dag-details" open={dev}>

        <summary>Stage DAG (reactflow) {dev ? "" : "— expand or use ?mode=dev"}</summary>

        <PipelineDagView pipeline={pipeline} mode={mode} />

      </details>

    </section>

  );

}

