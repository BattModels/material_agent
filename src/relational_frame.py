from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Relation:
    name: str
    child_table: str
    fk_in_child: str
    parent_pk: str
    # cardinality: str  # "one_to_one" or "one_to_many"
    local_key_in_child: Optional[str] = None


class RowView:
    """
    Single-row view.
      - row.col -> scalar
      - row.<relation> -> RelatedView
      - row.df -> 1-row DataFrame
    """
    def __init__(self, table: "Table", pk_value: Any):
        self._table = table
        self._pk_value = pk_value

    @property
    def table(self) -> "Table":
        return self._table

    @property
    def pk(self) -> Any:
        return self._pk_value

    @property
    def series(self) -> pd.Series:
        return self._table._get_row_series(self._pk_value)

    @property
    def df(self) -> pd.DataFrame:
        hits = self._table.df.loc[self._table.df[self._table.pk] == self._pk_value]
        if len(hits) != 1:
            raise KeyError(
                f"Table '{self._table.name}' expected 1 row where {self._table.pk}=={self._pk_value!r}, got {len(hits)}"
            )
        return hits.copy()

    def __getattr__(self, name: str) -> Any:
        if name in self._table.df.columns:
            return self.series[name]

        rel = self._table.relations.get(name)
        if rel is not None:
            return RelatedView(parent_row=self, relation=rel)

        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        # internal attributes must still work normally
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        # if it's a real column name, write-through to the backing DataFrame
        if name in self._table.df.columns:
            self._table.set_value(self._pk_value, name, value)
            return

        # prevent accidentally overwriting relations or random attributes
        if name in self._table.relations:
            raise AttributeError(f"'{name}' is a relation; assign via child table inserts/updates instead")

        raise AttributeError(f"Unknown attribute/column '{name}' on {self}")

    def __repr__(self) -> str:
        return f"<RowView {self._table.name}[{self._pk_value!r}]>"


class RelatedView:
    """
    Filtered child-table view for a specific parent row.
      - .df gives filtered DataFrame
      - [key] selects one child row (by local key if provided; else by child pk)
      - add_row(...) inserts into child and auto-fills FK
      - groupby / merge wrappers operate on .df
    """
    def __init__(self, parent_row: RowView, relation: Relation):
        self._parent_row = parent_row
        self._relation = relation

    @property
    def parent_relation(self) -> Relation:
        return self._relation
    
    @property
    def relations(self) -> Dict[str, Relation]:
        return self.child_table.relations

    @property
    def child_table(self) -> "Table":
        rf = self._parent_row.table.rf
        return rf.table(self._relation.child_table)

    @property
    def df(self) -> pd.DataFrame:
        parent_pk_val = self._parent_row.pk
        child = self.child_table
        return child.df.loc[child.df[self._relation.fk_in_child] == parent_pk_val]

    # wrappers
    def groupby(self, *args, **kwargs):
        return self.df.groupby(*args, **kwargs)

    def merge(self, other: Union["Table", "RelatedView", pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
        other_df = other.df if hasattr(other, "df") else other
        return self.df.merge(other_df, *args, **kwargs)

    # inserts (auto-fill FK)
    def add_row(
        self,
        row: Optional[Dict[str, Any]] = None,
        *,
        allow_update: bool = False,
        allow_new_columns: bool = False,
        **kwargs,
    ) -> RowView:
        data = {}
        if row:
            data.update(row)
        data.update(kwargs)

        # auto set FK -> parent pk value
        data[self._relation.fk_in_child] = self._parent_row.pk

        # delegate to child table
        return self.child_table.add_row(
            data,
            allow_update=allow_update,
            allow_new_columns=allow_new_columns,
        )

    def add_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        allow_update: bool = False,
        allow_new_columns: bool = False,
    ) -> List[RowView]:
        out = []
        for r in rows:
            out.append(
                self.add_row(
                    r,
                    allow_update=allow_update,
                    allow_new_columns=allow_new_columns,
                )
            )
        return out

    # row selection
    def __getitem__(self, key: Any) -> RowView:
        child = self.child_table
        sub = self.df

        if self._relation.local_key_in_child:
            col = self._relation.local_key_in_child
            hits = sub.loc[sub[col] == key]
            if len(hits) != 1:
                raise KeyError(
                    f"Expected exactly 1 row in related '{self._relation.name}' where {col}=={key!r}, got {len(hits)}"
                )
            pk_val = hits.iloc[0][child.pk]
            return child[pk_val]

        return child[key]

    def __iter__(self):
        child = self.child_table
        for pk_val in self.df[child.pk].tolist():
            yield child[pk_val]

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"<RelatedView {self._parent_row} -> {self._relation.name} ({len(self)} rows)>"


class Table:
    """
    DataFrame wrapper with PK, relations, and pandas-like wrappers.
      - table[pk] -> RowView
      - table.df -> underlying DataFrame
      - table.add_row / add_rows inserts after creation
      - table.groupby / merge wrappers forward to pandas
    """
    def __init__(self, rf: "RelationalFrame", name: str, df: pd.DataFrame, pk: str):
        self.rf = rf
        self.name = name
        self.pk = pk
        self.relations: Dict[str, Relation] = {}

        if pk not in df.columns:
            raise ValueError(f"Table '{name}' pk='{pk}' not found in columns: {list(df.columns)}")
        if df[pk].isna().any():
            raise ValueError(f"Table '{name}' pk column '{pk}' contains NaN")
        if df[pk].duplicated().any():
            dups = df.loc[df[pk].duplicated(), pk].tolist()[:10]
            raise ValueError(f"Table '{name}' pk '{pk}' must be unique; duplicates like: {dups}")

        self.df = df.reset_index(drop=True).copy()

    # relationship
    def add_relation(
        self,
        name: str,
        child_table: str,
        fk_in_child: str,
        # cardinality: str,
        local_key_in_child: Optional[str] = None,
    ) -> "Table":
        # if cardinality not in {"one_to_one", "one_to_many"}:
        #     raise ValueError("cardinality must be 'one_to_one' or 'one_to_many'")
        if name in self.relations:
            raise ValueError(f"Relation '{name}' already exists on table '{self.name}'")
        if local_key_in_child is None:
            local_key_in_child = self.rf.table(child_table).pk

        self.relations[name] = Relation(
            name=name,
            child_table=child_table,
            fk_in_child=fk_in_child,
            parent_pk=self.pk,
            # cardinality=cardinality,
            local_key_in_child=local_key_in_child,
        )
        return self
    
    def set_value(self, pk_value: Any, column: str, value: Any, *, allow_new_columns: bool = False) -> None:
        if column not in self.df.columns:
            if not allow_new_columns:
                raise AttributeError(f"Unknown column '{column}' on table '{self.name}'")
            self.df[column] = np.nan

        mask = (self.df[self.pk] == pk_value)
        if not mask.any():
            raise KeyError(f"Table '{self.name}' has no row with {self.pk}={pk_value!r}")

        self.df.loc[mask, column] = value

    # lookups
    def _get_row_series(self, pk_value: Any) -> pd.Series:
        hits = self.df.loc[self.df[self.pk] == pk_value]
        if len(hits) != 1:
            raise KeyError(f"Table '{self.name}' expected 1 row where {self.pk}=={pk_value!r}, got {len(hits)}")
        return hits.iloc[0]

    def __getitem__(self, pk_value: Any) -> RowView:
        # ensures row exists
        _ = self._get_row_series(pk_value)
        return RowView(self, pk_value)

    # inserts
    def add_row(
        self,
        row: Union[Dict[str, Any], pd.Series],
        *,
        allow_update: bool = False,
        allow_new_columns: bool = False,
    ) -> RowView:
        if isinstance(row, pd.Series):
            row = row.to_dict()
        row = dict(row)

        if self.pk not in row:
            raise ValueError(f"Missing primary key '{self.pk}' for table '{self.name}'")
        pk_val = row[self.pk]
        if pd.isna(pk_val):
            raise ValueError(f"Primary key '{self.pk}' cannot be NA for table '{self.name}'")

        exists_mask = (self.df[self.pk] == pk_val)
        exists = bool(exists_mask.any())

        # handle new columns
        incoming_cols = set(row.keys())
        existing_cols = set(self.df.columns)

        if not incoming_cols.issubset(existing_cols):
            if not allow_new_columns:
                extra = sorted(list(incoming_cols - existing_cols))
                raise ValueError(
                    f"Table '{self.name}' got unknown columns {extra}. "
                    f"Pass allow_new_columns=True to add them."
                )
            for c in sorted(incoming_cols - existing_cols):
                self.df[c] = np.nan

        # align to current schema (fill missing with NA)
        aligned = {c: row.get(c, np.nan) for c in self.df.columns}

        if exists:
            if not allow_update:
                raise ValueError(f"Table '{self.name}' already has pk={pk_val!r}. Use allow_update=True to update.")
            # update existing row in-place (only provided keys)
            for k, v in row.items():
                self.df.loc[exists_mask, k] = v
        else:
            new_row = pd.DataFrame({
                c: pd.Series([aligned[c]], dtype=self.df.dtypes[c])
                for c in self.df.columns
            })
            self.df = pd.concat([self.df, new_row], ignore_index=True)

        return self[pk_val]

    def add_rows(
        self,
        rows: List[Union[Dict[str, Any], pd.Series]],
        *,
        allow_update: bool = False,
        allow_new_columns: bool = False,
    ) -> List[RowView]:
        out = []
        for r in rows:
            out.append(
                self.add_row(
                    r,
                    allow_update=allow_update,
                    allow_new_columns=allow_new_columns,
                )
            )
        return out

    def update_row(self, pk_value: Any, **updates) -> RowView:
        return self.add_row({self.pk: pk_value, **updates}, allow_update=True, allow_new_columns=False)

    # wrappers
    def groupby(self, *args, **kwargs):
        return self.df.groupby(*args, **kwargs)

    def merge(self, other: Union["Table", RelatedView, pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
        other_df = other.df if hasattr(other, "df") else other
        return self.df.merge(other_df, *args, **kwargs)

    def sort_values(self, *args, **kwargs) -> pd.DataFrame:
        return self.df.sort_values(*args, **kwargs)

    def query(self, *args, **kwargs) -> pd.DataFrame:
        return self.df.query(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<Table {self.name} rows={len(self.df)} pk={self.pk}>"


class RelationalFrame:
    """
    Holds one root table + any number of child tables.
    """
    def __init__(self, root: str):
        self.root = root
        self._tables: Dict[str, Table] = {}
        


    def add_table(self, name: str, df: pd.DataFrame, pk: str) -> Table:
        if name in self._tables:
            raise ValueError(f"Table '{name}' already exists")
        t = Table(self, name=name, df=df, pk=pk)
        self._tables[name] = t
        return t

    def table(self, name: str) -> Table:
        if name not in self._tables:
            raise KeyError(f"Unknown table '{name}'. Known: {list(self._tables.keys())}")
        return self._tables[name]

    def __getattr__(self, name: str) -> Table:
        if name in self._tables:
            return self._tables[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def enforce_referential_integrity(self) -> None:
        """
        Optional sanity check you can call after inserts.
        Checks:
          - child FK values must exist in parent PK values
          - one_to_one requires FK uniqueness inside child table
        """
        for parent in self._tables.values():
            parent_pk_vals = set(parent.df[parent.pk].tolist())
            for rel in parent.relations.values():
                child = self.table(rel.child_table)

                # FK exists
                if rel.fk_in_child not in child.df.columns:
                    raise ValueError(
                        f"Relation '{parent.name}.{rel.name}' expects child column '{rel.fk_in_child}', "
                        f"but it does not exist in '{child.name}'."
                    )

                bad = child.df.loc[~child.df[rel.fk_in_child].isin(parent_pk_vals), rel.fk_in_child].unique().tolist()
                if bad:
                    raise ValueError(
                        f"Broken FK in relation '{parent.name}.{rel.name}': child '{child.name}.{rel.fk_in_child}' "
                        f"contains values not in parent '{parent.name}.{parent.pk}': {bad[:10]}"
                    )

                # if rel.cardinality == "one_to_one":
                #     if child.df[rel.fk_in_child].duplicated().any():
                #         dups = child.df.loc[child.df[rel.fk_in_child].duplicated(), rel.fk_in_child].tolist()[:10]
                #         raise ValueError(
                #             f"Relation '{parent.name}.{rel.name}' is one_to_one but child FK "
                #             f"'{child.name}.{rel.fk_in_child}' has duplicates like: {dups}"
                #         )
        
        print("Referential integrity check passed.")

    def flatten_explode(
        self,
        root: Optional[str] = None,
        drop_foreign_keys: bool = True,
    ) -> pd.DataFrame:
        """
        Recursively left-join all reachable relations starting from root.
        One-to-many relations naturally "explode" into multiple rows via merge.

        Column naming:
          - root columns: experiment.experiment_ID, experiment.note, ...
          - relation path: progress.step, progress.status, ...
          - deeper: progress.some_child.col, ...

        If you have multiple one-to-many branches at the same level, merges will create
        the cartesian product, which is the usual "explode one-to-many" behavior.
        """
        root_name = root or self.root
        root_table = self.table(root_name)
        df = self._flatten_table(table=root_table, path=root_name, visited=set())

        if drop_foreign_keys:
            # drop obvious FK columns that came from child sides when they match a parent join key,
            # but keep anything that's not clearly a join artifact.
            drop_cols = [c for c in df.columns if c.endswith(".") is False and False]  # no-op, kept simple
            _ = drop_cols  # placeholder (you can extend heuristics if you want)

        return df

    def _flatten_table(self, table: Table, path: str, visited: set) -> pd.DataFrame:
        # prevent cycles
        key = (table.name, path)
        if key in visited:
            return self._prefix_df(table.df, path)

        visited = set(visited)
        visited.add(key)

        base = self._prefix_df(table.df, path)

        # Merge each relation's flattened child into base
        for rel_name, rel in table.relations.items():
            child = self.table(rel.child_table)
            child_path = f"{rel_name}" if path == self.root else f"{path}.{rel_name}"

            child_flat = self._flatten_table(child, child_path, visited)

            left_on = f"{path}.{table.pk}"
            right_on = f"{child_path}.{rel.fk_in_child}"

            # one_to_one: still a merge; enforce uniqueness of fk in child if you want stricter checks
            base = base.merge(child_flat, how="left", left_on=left_on, right_on=right_on)

        return base

    @staticmethod
    def _prefix_df(df: pd.DataFrame, path: str) -> pd.DataFrame:
        out = df.copy()
        out.columns = [f"{path}.{c}" for c in out.columns]
        return out

    def __repr__(self) -> str:
        return f"<RelationalFrame root={self.root} tables={list(self._tables.keys())}>"
