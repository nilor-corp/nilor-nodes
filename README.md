# Nilor Nodes Documentation 👺

A collection of utility nodes for ComfyUI focusing on list manipulation, batch operations, and advanced I/O functionality.

## 🏭 Generators

<details>
<summary><b>Interpolated Float List</b></summary>

Generates a list of interpolated float values based on sections.

| Input | Type | Description |
|-------|------|-------------|
| number_of_floats | INT | Total number of float values to generate |
| number_of_sections | INT | Number of sections to divide into |
| section_number | INT | Current section being processed |
| interpolation_type | ["slinear", "quadratic", "cubic"] | Type of interpolation |

| Output | Type | Description |
|--------|------|-------------|
| floats | FLOAT | List of interpolated float values |

**Notes**: Creates smooth transitions between values using scipy's interpolation.
</details>

<details>
<summary><b>One Minus Float List</b></summary>

Creates an inverted list of float values (1 - x).

| Input | Type | Description |
|-------|------|-------------|
| list_of_floats | FLOAT | Input float list |

| Output | Type | Description |
|--------|------|-------------|
| floats | FLOAT | Inverted float values |

**Notes**: Simple inversion operation, useful for creating complementary values.
</details>

<details>
<summary><b>Remap Float List</b></summary>

Remaps a list of float values from one range to another.

| Input | Type | Description |
|-------|------|-------------|
| list_of_floats | FLOAT | Input float list |
| min_input | FLOAT | Minimum input value (default: 0.0) |
| max_input | FLOAT | Maximum input value (default: 1.0) |
| min_output | FLOAT | Minimum output value (default: 0.0) |
| max_output | FLOAT | Maximum output value (default: 1.0) |

| Output | Type | Description |
|--------|------|-------------|
| remapped_floats | FLOAT | Remapped float values |

**Notes**: Useful for scaling values between different ranges while preserving relationships.
</details>

<details>
<summary><b>Inverse Map Float List</b></summary>

Creates a mirror mapping of float values around their midpoint.

| Input | Type | Description |
|-------|------|-------------|
| list_of_floats | FLOAT | Input float list |

| Output | Type | Description |
|--------|------|-------------|
| floats | FLOAT | Inverse mapped values |

**Notes**: Automatically determines min/max from input list.
</details>

## 🛠️ Utilities

<details>
<summary><b>Int To List Of Bools</b></summary>

Converts an integer into a list of boolean values.

| Input | Type | Description |
|-------|------|-------------|
| number_of_images | INT | Number to convert |

| Output | Type | Description |
|--------|------|-------------|
| booleans | BOOLEAN | List of boolean values |

**Notes**: Creates a list where first N values are True, rest are False.
</details>

<details>
<summary><b>List of Ints</b></summary>

Generates a sequential or shuffled list of integers.

| Input | Type | Description |
|-------|------|-------------|
| min | INT | Starting integer (default: 0) |
| max | INT | Ending integer (default: 9) |
| shuffle | BOOLEAN | Whether to randomize order |

| Output | Type | Description |
|--------|------|-------------|
| ints | INT | List of integers |

**Notes**: Output is always a list, even for single values.
</details>

<details>
<summary><b>Select Index From List</b></summary>

Extracts a single item from a list at the specified index.

| Input | Type | Description |
|-------|------|-------------|
| list_of_any | any | Input list of any type |
| index | INT | Index to select (default: 0) |

| Output | Type | Description |
|--------|------|-------------|
| any | any | Selected item |

**Notes**: Uses custom AnyType to accept any input type. Handles tensor unpacking automatically.
</details>

<details>
<summary><b>Shuffle Image Batch</b></summary>

Randomly reorders images in a batch.

| Input | Type | Description |
|-------|------|-------------|
| images | IMAGE | Batch of images |
| seed | INT | Random seed for shuffling |

| Output | Type | Description |
|--------|------|-------------|
| images | IMAGE | Shuffled image batch |

**Notes**: Maintains batch dimensions while randomizing order.
</details>

## 💾 I/O Operations

<details>
<summary><b>Save Image To HF Dataset</b></summary>

Uploads images to a HuggingFace dataset.

| Input | Type | Description |
|-------|------|-------------|
| image | IMAGE | Image to upload |
| repository_id | STRING | HuggingFace dataset repository |
| hf_auth_token | STRING | HuggingFace authentication token |
| filename_prefix | STRING | Prefix for saved files |

**Notes**: Requires HuggingFace authentication token and repository access.
</details>

<details>
<summary><b>Save EXR Arbitrary</b></summary>

Saves multi-channel data as an OpenEXR file.

| Input | Type | Description |
|-------|------|-------------|
| channels | any | List of tensor channels |
| filename_prefix | STRING | Output filename prefix |

**Notes**: Supports arbitrary number of channels. Each channel must have same dimensions.
</details>

<details>
<summary><b>Save Video To HF Dataset</b></summary>

Uploads video files to a HuggingFace dataset.

| Input | Type | Description |
|-------|------|-------------|
| filenames | VHS_FILENAMES | List of video files |
| repository_id | STRING | HuggingFace dataset repository |
| hf_auth_token | STRING | HuggingFace authentication token |
| filename_prefix | STRING | Prefix for saved files |

**Notes**: Handles batch upload of multiple video files.
</details>

## 📡 Core Nilor Services

<details>
<summary><b>Worker Consumer Service</b></summary>

The `worker_consumer.py` script is a background service that runs on each ComfyUI worker. It is responsible for pulling jobs from the central ElasticMQ `jobs_to_process` queue and submitting them to its local ComfyUI instance for processing. This service is essential for the distributed architecture of the system.

**Key Responsibilities:**
-   Continuously polls the `jobs_to_process` queue for new jobs using long polling.
-   When a job is received, it extracts the workflow data and submits it to the local ComfyUI server.
-   Deletes the job message from the queue upon successful submission to prevent reprocessing.
-   If submission fails, the message remains on the queue to be picked up by another worker.

</details>

<details>
<summary><b>Environment Variables</b></summary>

The `nilor-nodes` require a `.env` file to be present in the `ComfyUI` directory to configure the connection to the core services (MinIO, ElasticMQ, and the Brain API). To set it up, create a file named `.env` in the root of your `ComfyUI` directory by copying the `.env.example` template.

**Instructions:**
1.  Create a new file named `.env` in the `ComfyUI` directory.
2.  Copy the contents of the `.env.example` file into your new `.env` file.
3.  Replace the placeholder values with your actual credentials and endpoint URLs for your local or production environment.

</details>