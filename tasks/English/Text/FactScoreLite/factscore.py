import numpy as np
from . import FactScorer, AtomicFactGenerator
from .state_handler import StateHandler
from . import configs
from tqdm.std import tqdm


class FactScore:
    def __init__(self, gamma: int = 10):
        self.atomic_fact_generator = AtomicFactGenerator()
        self.fact_scorer = FactScorer()
        self.facts_handler = StateHandler(configs.facts_db_path)
        self.decisions_handler = StateHandler(configs.decisions_db_path)
        self.gamma = gamma

    def get_facts(self, generations: list) -> list:
        """
        Extract facts from a list of generations using AtomicFactGenerator.
        Saves the results in a json file using the StateHandler.
        """
        print("Extracting facts from generations...")

        generation_facts_pairs = self.facts_handler.load()

        for generation in tqdm(generations[len(generation_facts_pairs) :]):
            atomic_facts_of_generation = self.atomic_fact_generator.run(generation)
            atomic_facts_of_generation = [
                fact
                for sentence, atomic_facts in atomic_facts_of_generation
                for fact in atomic_facts
            ]
            generation_facts_pairs.append(
                {"generation": generation, "facts": atomic_facts_of_generation}
            )
            self.facts_handler.save(generation_facts_pairs)

        assert len(generation_facts_pairs) == len(
            generations
        ), "Number of generations and generation-facts pairs must match."

        return generation_facts_pairs

    def calculate_score(self, decision: list) -> tuple:
        """
        Calculates the score of a generation based on whether its facts are supported.
        Prevents ZeroDivisionError by handling empty decision lists.
        """
        if len(decision) == 0:
            return 0, 0 

        score = np.mean([d["is_supported"] for d in decision])
        init_score = score

        if self.gamma:
            penalty = (
                1.0 if len(decision) >= self.gamma else np.exp(1 - self.gamma / max(1, len(decision)))
            )
            score = penalty * score

        return score, init_score

    def get_decisions(self, generation_facts_pairs: list, knowledge_sources: list) -> list:
        """
        Scores the facts related to each generation based on the knowledge source.
        """
        print("Generating decisions...")

        if not generation_facts_pairs or not knowledge_sources:
            print("Warning: Empty generation_facts_pairs or knowledge_sources!")
            return [], []

        decisions = self.decisions_handler.load()
        scores = []
        init_scores = []

        for entry in decisions:
            score, init_score = self.calculate_score(entry["decision"])
            init_scores.append(init_score)
            scores.append(score)

        assert len(generation_facts_pairs) == len(
            knowledge_sources
        ), "Number of generation-facts pairs and knowledge sources should be the same."

        current_index = len(decisions)

        for entry, knowledge_source in tqdm(
            zip(
                generation_facts_pairs[current_index:],
                knowledge_sources[current_index:],
            )
        ):
            generation, facts = entry["generation"], entry["facts"]

            if not facts:
                print(f"Warning: No facts extracted for generation: {generation}")
                decisions.append({"generation": generation, "decision": []})
                scores.append(0)
                init_scores.append(0)
                continue

            decision = self.fact_scorer.get_score(facts, knowledge_source)
            score, init_score = self.calculate_score(decision)

            init_scores.append(init_score)
            scores.append(score)
            decisions.append({"generation": generation, "decision": decision})
            self.decisions_handler.save(decisions)

            assert len(facts) == len(
                decision
            ), "Number of facts and decisions for that generation should be the same."

        assert len(decisions) == len(
            generation_facts_pairs
        ), "Number of decisions and generation-facts pairs should be the same."

        return scores, init_scores

    def get_factscore(self, generations: list, knowledge_sources: list) -> tuple:
        """
        Extracts atomic facts from generations and scores them.
        """
        assert len(generations) == len(
            knowledge_sources
        ), "`generations` and `knowledge_sources` should have the same length."

        facts = self.get_facts(generations)
        scores, init_scores = self.get_decisions(facts, knowledge_sources)

        if not scores: 
            return 0.0, 0.0

        return np.mean(scores), np.mean(init_scores)
