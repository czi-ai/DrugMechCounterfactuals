"""
Entities
"""

from collections import defaultdict, Counter
import dataclasses
import re
from typing import Any

from drugmechcf.llmx.text import normalize_name, standardize_name, parse_standardize_name, NormalizedFirstIn


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

DEBUG = False

PATT_NM_FOR_NM = re.compile(r"([^(]+)\s+\(for ([^)]+)\)")


# -----------------------------------------------------------------------------
#   Classes: entity types
# -----------------------------------------------------------------------------

# Allow a hash func to be created, even though the class is mutable: (unsafe_hash=True, and every field has hash=False)
# This will allow Entity objs to be dict-key (used in EntityCollator).
@dataclasses.dataclass(unsafe_hash=True)
class Entity:
    name: str = dataclasses.field(hash=False)
    entity_type: str = dataclasses.field(hash=False)

    alt_names: list[str] = dataclasses.field(default_factory=list, hash=False)

    target_entity_type: str = dataclasses.field(default=None, hash=False)
    target_entity_name: str = dataclasses.field(default=None, hash=False)

    interaction_descr: str = dataclasses.field(default=None, hash=False)

    def add_alt_name(self, new_name):
        if new_name in self.alt_names:
            return
        self.alt_names.append(new_name)
        return

    def asdict(self) -> dict[str, Any]:
        # noinspection PyTypeChecker
        return dataclasses.asdict(self)

    def short_str(self):
        msg = f"{self.entity_type}: {self.name} [alt: {', '.join(self.alt_names)}]"
        return msg

    @staticmethod
    def from_dict(adict: dict[str, Any]) -> "Entity":
        entt = Entity(**adict)
        return entt
# /


@dataclasses.dataclass(unsafe_hash=True)
class DrugOrFamily(Entity):

    examples: list[str] = dataclasses.field(default_factory=list, hash=False)

    approved_for: list[str] = dataclasses.field(default_factory=list, hash=False)

    entity_type: str = dataclasses.field(default="drug", hash=False)

    @property
    def mechanism(self):
        """Synonym for `interaction_descr`"""
        return self.interaction_descr

    @mechanism.setter
    def mechanism(self, value: str):
        self.interaction_descr = value

    def add_example(self, new_example):
        if new_example in self.examples or new_example in self.alt_names:
            return
        self.examples.append(new_example)
        return

    def is_family(self):
        return self.examples is not None and len(self.examples) > 1

    @staticmethod
    def from_dict(adict: dict[str, Any]) -> "DrugOrFamily":
        adict = adict.copy()
        alt_names = adict.pop('alt_names', [])
        examples = adict.pop('examples', [])

        entt = DrugOrFamily(**adict)

        uniq_alt_names = NormalizedFirstIn(entt.name)
        uniq_alt_names.add_from(alt_names)
        entt.alt_names = uniq_alt_names.get_values_list()

        uniq_examples = NormalizedFirstIn(entt.name)
        uniq_examples.exclude_from(alt_names)
        uniq_examples.add_from(examples)
        entt.examples = uniq_examples.get_values_list()

        return entt

# /


@dataclasses.dataclass
class Drug(Entity):

    entity_type: str = dataclasses.field(default="drug", hash=False)

    approved_for: set[str] = dataclasses.field(default_factory=set, hash=False)

    families: set[str] = dataclasses.field(default_factory=set, hash=False)

    interacts_with: list[Any] = dataclasses.field(default_factory=list, hash=False)

    def short_str(self):
        msg = super().short_str()
        msg += f"; families: [{', '.join(self.families)}]"
        return msg

    def asdict(self) -> dict[str, Any]:
        # noinspection PyTypeChecker
        d = dataclasses.asdict(self)

        # Convert Set to List
        d["approved_for"] = list(self.approved_for)
        d["families"] = list(self.families)

        return d

    @staticmethod
    def from_dict(adict: dict[str, Any]) -> "Drug":

        # Ensure values are Set's
        for k in ["approved_for", "families"]:
            if v := adict.get(k):
                adict[k] = set(v)

        drug = Drug(**adict)

        return drug
# /


# Allow a hash func to be created, even though the class is mutable.
# This will allow InteractionInfo objs to be identified by ptr in lists, sets (used in EntityCollator).
@dataclasses.dataclass(unsafe_hash=True)
class InteractionInfo:
    """
    Represents an edge between two entities.
    """

    src_entity: Entity = dataclasses.field(hash=False)
    tgt_entity: Entity = dataclasses.field(hash=False)

    causal_direction: str = dataclasses.field(default=None, hash=False)
    """The prompt `OPTS_CAUSAL_DIRECTION_TYPED` defines values: BACK, FORWARD, UNKNOWN"""

    direction_of_effect: str = dataclasses.field(default=None, hash=False)

    causal_direction_descr: str = dataclasses.field(default=None, hash=False)
    """The prompt `OPTS_DIRECTION_OF_EFFECT_TYPED` defines values: NON-INVERTING, INVERTING, UNKNOWN"""

    direction_of_effect_descr: str = dataclasses.field(default=None, hash=False)

    def causal_direction_is_unknown(self):
        return self.causal_direction in [None, "UNKNOWN"]

    def direction_of_effect_is_unknown(self):
        return self.direction_of_effect in [None, "UNKNOWN"]

    def asdict(self) -> dict[str, Any]:
        # noinspection PyTypeChecker
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(adict: dict[str, Any]) -> "InteractionInfo":
        adict = adict.copy()
        adict["src_entity"] = Entity.from_dict(adict["src_entity"])
        adict["tgt_entity"] = Entity.from_dict(adict["tgt_entity"])

        obj = InteractionInfo(**adict)
        return obj
# /


# -----------------------------------------------------------------------------
#   Classes: EntityCollator
# -----------------------------------------------------------------------------


class EntityCollator:

    def __init__(self):
        self.entity_type = None
        self._reset_indices()
        return

    def _reset_indices(self):
        self.entities = []
        self.entity_type = None

        # normalized-name => set[int]
        self.normzd_name_idx = defaultdict(set)
        self.normzd_alt_name_idx = defaultdict(set)
        self.normlzd_family_idx = defaultdict(set)

        # Interactions
        self._reset_interactions()

        return

    def _reset_interactions(self):
        self.interactions: list[InteractionInfo] = []
        self.srcentt_2_ixns: dict[Entity, list[InteractionInfo]] = defaultdict(list)
        self.tgtentt_2_ixns: dict[Entity, list[InteractionInfo]] = defaultdict(list)
        return

    # .......................................................................................
    #   Handling of Interactions, if needed
    # .......................................................................................

    # noinspection PyAttributeOutsideInit
    def set_interactions(self, interactions: list[InteractionInfo]):
        """
        Sets the interactions indices
        """
        self._reset_interactions()

        self.interactions = interactions
        for ixn in self.interactions:
            self.srcentt_2_ixns[ixn.src_entity].append(ixn)
            self.tgtentt_2_ixns[ixn.tgt_entity].append(ixn)

        return

    def add_interaction(self, ixn: InteractionInfo):
        self.interactions.append(ixn)
        self.srcentt_2_ixns[ixn.src_entity].append(ixn)
        self.tgtentt_2_ixns[ixn.tgt_entity].append(ixn)
        return

    def update_interaction(self, old_entt: Entity, new_entt: Entity,
                           add_new_ixn: bool = False):
        """
        Replaces pointers to `old_entt` with `new_entt` in any ixns.
        IF add_new_ixn THEN a copy of the existing ixn is first made.
        Interaction Indices are updated.

        NoOp if no interactions.
        """
        if not self.interactions:
            return

        added_ixns = []

        # Handle interactions set in `self.set_interactions():
        #   Any pointers to `new_entt` will change to `old_entt`

        for ixn in (matches := self.srcentt_2_ixns[old_entt]):
            if add_new_ixn:
                # create a copy
                ixn = dataclasses.replace(ixn)
                added_ixns.append(ixn)
            else:
                matches.remove(ixn)
                self.srcentt_2_ixns[new_entt].append(ixn)

            ixn.src_entity = new_entt

        for ixn in (matches := self.tgtentt_2_ixns[old_entt]):
            # create a copy, if not already created
            if add_new_ixn and ixn not in added_ixns:
                ixn = dataclasses.replace(ixn)
                added_ixns.append(ixn)
            else:
                matches.remove(ixn)
                self.tgtentt_2_ixns[new_entt].append(ixn)

            ixn.tgt_entity = new_entt

        for ixn in added_ixns:
            self.add_interaction(ixn)

        return

    def get_all_interactions(self) -> list[InteractionInfo]:
        return self.interactions

    # .......................................................................................
    #   Handling of Entities
    # .......................................................................................

    def add_new_entity(self, new_entt: Entity | Drug):
        if self.entity_type is None:
            self.entity_type = type(new_entt)
        else:
            assert isinstance(new_entt, self.entity_type), f'All entities must be of type `{self.entity_type}`'

        self.entities.append(new_entt)
        new_entt_idx = len(self.entities) - 1

        self._add_name_index(new_entt.name, new_entt_idx, self.normzd_name_idx)

        for nm in new_entt.alt_names:
            self._add_name_index(nm, new_entt_idx, self.normzd_alt_name_idx)

        if isinstance(new_entt, Drug):
            for nm in new_entt.families:
                self._add_name_index(nm, new_entt_idx, self.normlzd_family_idx)

        return

    def add_new_entities(self, entities: list[Entity] | list[Drug]):
        for entt in entities:
            self.add_new_entity(entt)
        return

    def get_all_entities(self) -> list[Entity]:
        return self.entities

    @staticmethod
    def _add_name_index(name: str, idx: int, normzd_name_index: dict[str, set[int]]):
        normzd_nm = normalize_name(name)
        normzd_name_index[normzd_nm].add(idx)
        return

    @staticmethod
    def _del_name_index(name: str, idx: int, normzd_name_index: dict[str, set[int]]):
        normzd_nm = normalize_name(name)
        normzd_name_index[normzd_nm].discard(idx)
        return

    def merge_entities(self, old_entt: Entity | Drug, new_entt: Entity | Drug):
        """
        Merges info from `new_entt` into existing entt `tgt_entt`.
        """

        assert type(old_entt) is type(new_entt), "`tgt_entt` and `new_entt` must be of the same type."

        tgt_entt_idx = self.entities.index(old_entt)

        if tgt_entt_idx < 0:
            print(f"*** Warning: {tgt_entt_idx=} < 0, for tgt_entt {old_entt.name}")

        if DEBUG:
            print("Merging:")
            print("   old =", old_entt)
            print("   new =", new_entt)
            print()

        # TODO: Dont use new-name if it is a Family name.
        # Use new_drug.name as the main name?
        if len(old_entt.name) < 7 and len(new_entt.name) > len(old_entt.name) + 2:
            if DEBUG:
                print(f"  Switching names: {old_entt.name} <=> {new_entt.name}")

            # Set tgt_entt's name, and make old name an alt-name
            old_entt.name, new_entt.name = new_entt.name, old_entt.name
            # Remove old name from name-index
            self._del_name_index(new_entt.name, tgt_entt_idx, self.normzd_name_idx)

        # Add and index new alt-names
        uniq_names = NormalizedFirstIn(old_entt.name)
        uniq_names.update_from(old_entt.alt_names, [new_entt.name] + new_entt.alt_names)
        old_entt.alt_names.extend(uniq_names.get_values())
        for nm in uniq_names.get_values():
            self._add_name_index(nm, tgt_entt_idx, self.normzd_alt_name_idx)

        # Add and index: approved_for, families
        if isinstance(old_entt, Drug):
            uniq_names = NormalizedFirstIn(None)
            uniq_names.update_from(old_entt.approved_for, new_entt.approved_for)
            old_entt.approved_for.update(uniq_names.get_values())

            uniq_names = NormalizedFirstIn(None)
            uniq_names.update_from(old_entt.families, new_entt.families)
            old_entt.families.update(uniq_names.get_values())
            for nm in uniq_names.get_values():
                self._add_name_index(nm, tgt_entt_idx, self.normlzd_family_idx)

        if DEBUG:
            print("   New tgt =", old_entt)
            print("---------")
            print()

        return

    @staticmethod
    def _get_substr_matches(normzd_nm: str, index: dict[str, set[int]]) -> set[int]:
        matched_idxs = set()
        for k, v in index.items():
            if normzd_nm in k:
                matched_idxs.update(v)

        return matched_idxs

    def find_name_matches(self, name: str, as_substring=False, also_check_families=False) -> list[Drug]:
        matched_idxs = set()
        normzd_nm = normalize_name(name)

        # ---
        def get_idxs(nm, index):
            if as_substring:
                return self._get_substr_matches(nm, index)
            else:
                return index.get(nm)
        # ---

        if mentt_idxs := get_idxs(normzd_nm, self.normzd_name_idx):
            matched_idxs.update(mentt_idxs)

        if mentt_idxs := get_idxs(normzd_nm, self.normzd_alt_name_idx):
            matched_idxs.update(mentt_idxs)

        if also_check_families:
            if mentt_idxs := get_idxs(normzd_nm, self.normlzd_family_idx):
                matched_idxs.update(mentt_idxs)

        return [self.entities[i] for i in matched_idxs]

    def find_matching_entities(self, new_entity: Entity | Drug) -> list[Drug]:
        """
        Matches on entt.name, entt.alt_names
        """
        matched_idxs = set()
        for nm in [new_entity.name] + new_entity.alt_names:
            normzd_nm = normalize_name(nm)
            if mdrugs := self.normzd_name_idx.get(normzd_nm):
                matched_idxs.update(mdrugs)
            if mdrugs := self.normzd_alt_name_idx.get(normzd_nm):
                matched_idxs.update(mdrugs)

        return [self.entities[i] for i in matched_idxs]

    def merge_index_entities(self, entities: list[Entity] | list[Drug]):
        for new_entt in entities:
            matches = self.find_matching_entities(new_entt)

            if matches:
                if len(matches) > 1 or DEBUG:
                    print(f"*** {len(matches)} matches found for {new_entt.short_str()}")
                    for n, mentt in enumerate(matches, start=1):
                        print(f"  [{n}] {mentt.short_str()}")
                    print("  ------\n")

                for n, m_entt in enumerate(matches):
                    self.merge_entities(m_entt, new_entt)
                    # For matches after the 1st match, create new ixn.
                    # This allows same edge to be copied.
                    self.update_interaction(new_entt, m_entt, add_new_ixn=n > 0)

            else:
                if DEBUG:
                    print()
                    print(f"** Adding {new_entt.short_str()}")

                self.add_new_entity(new_entt)

        return

    # .......................................................................................
    #   Collate Drug's from DrugOrFamily's
    # .......................................................................................

    def collate_drugs_from_drugs_or_families(self, drugs_or_families: list[DrugOrFamily]) -> list[Drug]:

        self._reset_indices()

        for dr_or_fam in drugs_or_families:
            main_name, alt_names, examples = parse_standardize_name(dr_or_fam.name)

            new_drugs = []

            uniq_examples = NormalizedFirstIn(main_name)
            uniq_examples.update_from(alt_names, examples)
            uniq_examples.add_from([standardize_name(nm) for nm in dr_or_fam.examples])

            examples = uniq_examples.get_values_list()

            uniq_alt_names = NormalizedFirstIn(main_name)
            uniq_alt_names.add_from(alt_names)
            uniq_alt_names.add_from([standardize_name(nm) for nm in dr_or_fam.alt_names])

            alt_names = uniq_alt_names.get_values_list()

            # Heuristic: Any v. long example names should be alt-names
            short_examples, long_examples = separate_long_multitoken_names(examples)
            if long_examples:
                examples = short_examples
                alt_names.extend(long_examples)

            uniq_approved_for = NormalizedFirstIn()
            uniq_approved_for.add_from([standardize_name(nm) for nm in dr_or_fam.approved_for])
            approved_for = uniq_approved_for.get_values()

            # IF there are any examples THEN this is a Drug-Family

            if not examples:
                drug = Drug(name=main_name, alt_names=alt_names, approved_for=approved_for)

                new_drugs.append(drug)

            else:
                uniq_nms, uniq_exs = separate_altnames_examples(alt_names)
                family_nms = [main_name] + uniq_nms
                examples = examples + uniq_exs

                # if DEBUG:
                #     print()
                #     print(f"collate_drugs_from_drugs_or_families: Family =", ", ".join(family_nms))
                #     print(f"    Examples =", ", ".join(examples))

                for nm in examples:
                    nm_main_name, nm_alt_names, nm_examples = parse_standardize_name(nm)

                    drug = Drug(name=nm_main_name, alt_names=nm_alt_names,
                                families=set(family_nms), approved_for=approved_for)
                    new_drugs.append(drug)

            # Replace dr_or_fam with new_drugs in interactions
            if self.interactions:
                for n, new_drug_ in enumerate(new_drugs):
                    self.update_interaction(dr_or_fam, new_drug_, add_new_ixn=n > 0)

            self.merge_index_entities(new_drugs)

        return self.entities
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def collate_entities_from_dicts(entity_type: str,
                                entity_dicts: list[dict[str, Any]],
                                tgt_entity: Entity) -> tuple[list[Entity], dict[str, int]]:
    """
    Merges entities with the same normalized name.

    :param entity_type: The entity type for all the entities
    :param entity_dicts: List of dicts, with fields defined in prompts:
        - name: the name of the {src_type}.
        - alt_names: a list of any alternative names for this {src_type}.
        - interaction: a brief description of this {src_type}'s interaction with {tgt_entity}.
    :param tgt_entity: The target entity that each entity here interacts with

    :return: Collated list, uniq'ed from `entity_dicts`
    """
    name2ent: dict[str, Entity | DrugOrFamily] = dict()

    normzd_name_counts = Counter()

    if not entity_dicts:
        return [], normzd_name_counts

    for edict in entity_dicts:
        alt_names = edict.get("alt_names", [])

        normalized_name = normalize_name(edict["name"])
        normzd_name_counts.update([normalized_name])

        if entt := name2ent.get(normalized_name):
            alt_names.append(edict["name"])

        else:
            if entity_type == "drug":
                entt = DrugOrFamily(name = edict["name"], entity_type=entity_type,
                                    target_entity_type=tgt_entity.entity_type,
                                    target_entity_name=tgt_entity.name,
                                    interaction_descr=edict["mechanism"],
                                    )
            else:
                entt = Entity(name = edict["name"], entity_type=entity_type,
                              target_entity_type=tgt_entity.entity_type,
                              target_entity_name=tgt_entity.name,
                              interaction_descr=edict["interaction"])
            name2ent[normalized_name] = entt

        if alt_names:
            uniq_alt_names = NormalizedFirstIn(entt.name)
            uniq_alt_names.update_from(entt.alt_names, alt_names)
            entt.alt_names.extend(uniq_alt_names.get_values_list())

        if entity_type == "drug" and (examples := edict.get("examples", [])):
            uniq_examples = NormalizedFirstIn()
            uniq_examples.update_from(entt.examples, examples)
            entt.examples.extend(uniq_examples.get_values_list())

    return list(name2ent.values()), normzd_name_counts


def separate_altnames_examples(names: list[str],
                               ) -> tuple[list[str], list[str]]:
    """
    Sometimes an alt_name is actually mentioning an example, in the form 'Sirolimus (for Rapamycin)'.
    Uniqueness only determined using simple set.

    :return: uniq_names, uniq_examples
    """

    uniq_examples = set()
    uniq_names = set()

    for aname in names:
        if aname.endswith(")") and (m := PATT_NM_FOR_NM.match(aname)):
            uniq_examples.add(m.group(1))
            uniq_examples.update([s.strip() for s in m.group(2).split(",")])
        else:
            uniq_names.add(aname)

    return sorted(uniq_names), sorted(uniq_examples)


def separate_long_multitoken_names(names: list[str],
                                   min_length: int = 50, min_spaces: int = 3) -> tuple[list[str], list[str]]:
    """
    :return: shorter_names, v-long-names
    """
    short_names = []
    long_names = []
    for nm in names:
        if len(nm) >= min_length and nm.count(' ') >= min_spaces:
            long_names.append(nm)
        else:
            short_names.append(nm)

    return short_names, long_names
