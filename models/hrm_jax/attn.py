import jax.experimental.pallas.ops.tpu.flash_attention

# jax.experimental.pallas.ops.tpu.flash_attention.flash_attention(
#     q, k, v, ab=ab
# )

    # @named_scope("eqx.nn.MultiheadAttention")
    # def __call__(
    #     self,
    #     query: Float[Array, "q_seq q_size"],
    #     key_: Float[Array, "kv_seq k_size"],
    #     value: Float[Array, "kv_seq v_size"],
    #     mask: None | _Mask = None,
    #     *,
    #     key: PRNGKeyArray | None = None,
    #     inference: bool | None = None,
    #     deterministic: bool | None = None,
    #     process_heads: None | _ProcessHeads = None,
    # ) -> Float[Array, "q_seq o_size"]:
    #     """**Arguments:**

    #     - `query`: Query embedding. Should be a JAX array of shape
    #         `(query_seq_length, query_size)`.
    #     - `key_`: Key embedding. Should be a JAX array of shape
    #         `(kv_seq_length, key_size)`.
    #     - `value`: Value embedding. Should be a JAX array of shape
    #         `(kv_seq_length, value_size)`.
    #     - `mask`: Optional mask preventing attention to certain positions. Should either
    #         be a JAX array of shape `(query_seq_length, kv_seq_length)`, or (for custom
    #         per-head masking) `(num_heads, query_seq_length, kv_seq_length)`. A value of
    #         `False` at a position indicates that position should be ignored.
    #     - `key`: A `jax.random.PRNGKey` used for dropout. Unused if `dropout = 0`.
    #         (Keyword only argument.)
    #     - `inference`: As [`equinox.nn.Dropout.__call__`][]. (Keyword only
    #         argument.)
    #     - `deterministic`: (Deprecated in favour of `inference`.)
    #     - `process_heads`: A function that takes in the query, key, and value heads and
    #         returns new query, key, and value heads. For example, this can be
    #         used to implement relative positional embeddings -
    #         see e.g. `RotaryPositionalEmbedding`for an example. (Keyword only argument.)

    #     **Returns:**

    #     A JAX array of shape `(query_seq_length, output_size)`.
    #     """

    #     if deterministic is not None:
    #         inference = deterministic
    #         warnings.warn(
    #             "MultiheadAttention()(deterministic=...) is deprecated "
    #             "in favour of MultiheadAttention()(inference=...)"
    #         )

    #     query_seq_length, _ = query.shape
    #     kv_seq_length, _ = key_.shape
    #     kv_seq_length2, _ = value.shape
    #     if kv_seq_length != kv_seq_length2:
    #         # query length can be different
    #         raise ValueError("key and value must both be sequences of equal length.")

    #     query_heads = self._project(self.query_proj, query)
    #     key_heads = self._project(self.key_proj, key_)
    #     value_heads = self._project(self.value_proj, value)

    #     if process_heads is not None:
    #         q_shape, k_shape, v_shape = (
    #             query_heads.shape,
    #             key_heads.shape,
    #             value_heads.shape,
    #         )
    #         query_heads, key_heads, value_heads = process_heads(
    #             query_heads, key_heads, value_heads
    #         )

    #         if (
    #             query_heads.shape != q_shape
    #             or key_heads.shape != k_shape
    #             or value_heads.shape != v_shape
    #         ):
    #             raise ValueError(
    #                 "process_heads must not change the shape of the heads."
    #             )

    #     attn_fn = partial(
    #         dot_product_attention, dropout=self.dropout, inference=inference
    #     )
    #     keys = None if key is None else jax.random.split(key, query_heads.shape[1])
    #     if mask is not None and mask.ndim == 3:
    #         # Batch `mask` and `keys` down their 0-th dimension.
    #         attn = jax.vmap(attn_fn, in_axes=1, out_axes=1)(
    #             query_heads, key_heads, value_heads, mask=mask, key=keys
    #         )
    #     else:
    #         # Batch `keys` down its 0-th dimension.
    #         attn = jax.vmap(ft.partial(attn_fn, mask=mask), in_axes=1, out_axes=1)(
    #             query_heads, key_heads, value_heads, key=keys
    #         )
    #     attn = attn.reshape(query_seq_length, -1)

    #     return jax.vmap(self.output_proj)(attn)
    
            # jax.experimental.shard_map.shard_map((
            #     lambda q, k, v, ab: jax.experimental.pallas.ops.tpu.flash_attention.flash_attention(
            #         q, k, v, ab=ab
            #     )
            # ),
            # mesh=mesh,
            # # TODO what to do with batch?
            # in_specs=(P(axis_name_to_mesh_name.get("batch"),
            #     axis_name_to_mesh_name.get(self.head_axis),
            #     axis_name_to_mesh_name.get(self.seq_axis),
            #     # splitting apart the head dimension is just silly
            #     None),) * 3
            # + (P(axis_name_to_mesh_name.get("batch"),
            #      axis_name_to_mesh_name.get(self.head_axis),
            #      # no KV seq split 😔
            #      None,
            #      axis_name_to_mesh_name.get(self.seq_axis)
            #      ),),
            # out_specs=P(axis_name_to_mesh_name.get("batch"),
            #    axis_name_to_mesh_name.get(self.head_axis),
            #    axis_name_to_mesh_name.get(self.seq_axis),
            #    None),
            # check_rep=False)(q, k, v, ab),