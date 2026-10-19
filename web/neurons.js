// The sixteen neurons in words, one home for the story's field guide and the explorer's panel. The facts behind each
// sentence: research/findings/neurons.md and circuit.md. Numbers stay out of the prose; the pages compute them beside it.
window.NEURONS = {
  2: { name: 'The record keeper', text: 'Holds the largest number seen so far, scaled up and one step behind. Each time a record falls it adds the jump from n4; otherwise it keeps its own value, and all four comparators measure against it.' },
  4: { name: 'The jump', text: 'Fires only when a new number beats the record, by how far it beats it. It hands the jump to n2 and wakes the timers; its own vote is a veto on position 9, because a record set by the last number cannot be the runner-up.' },
  7: { name: 'How far below the record', text: 'Measures how far each new number lands below the record. The comparators n8 and n1 subtract it, and it watches position 9: a runner-up arriving last lands just below the record, which switches n7 down and lifts its veto on 9.' },
  6: { name: 'Below the record, without the last number', text: 'Like n7, but measured against the record with the previous number left out, as if it had never arrived. It drives the comparator n1 and the timers n14 and n0.' },
  1: { name: 'Below the record, without the one before', text: 'Like n6, but with the number two back left out instead: it adds n6 and subtracts n7. It drives the comparator n8 and the timers n12 and n0.' },
  8: { name: 'Below the runner-up so far', text: 'Measures how far each new number lands below the runner-up seen so far; of the six, its description fits the loosest. It reads n7, n2 and n1, and holds down the timers n3, n15 and n0.' },
  14: { name: 'Watches position 0', text: 'Votes against position 0, unless the runner-up arrives there and switches it down, lifting the veto. n6 drives it.' },
  15: { name: 'Votes for positions 1 and 2', text: 'A runner-up arriving at position 1 or 2 switches it up, adding a vote for that position. n10 and n8 hold it down.' },
  10: { name: 'Watches positions 2 and 3', text: 'Votes against positions 2 and 3, unless the runner-up arrives at one of them and switches it down. n11 drives it, and it holds n15 down.' },
  11: { name: 'Watches position 4', text: 'Votes against position 4, unless the runner-up arrives there and switches it down. n12 and n7 drive it, and it passes the signal on to n10.' },
  12: { name: 'Watches positions 5 and 6', text: 'Votes against positions 5 and 6, unless the runner-up arrives at one of them and switches it down. n1 drives it, and it drives n11.' },
  0: { name: 'Watches position 7', text: 'Votes against position 7, unless the runner-up arrives there and switches it down. n6 drives it, n7 holds it down, and it holds n3 down.' },
  5: { name: "Position 8's own neuron", text: 'Fires when the record or the runner-up arrives at position 8, and votes for position 8: the one position the clock would confuse with position 0.' },
  3: { name: 'A faint helper', text: 'Adds small votes for positions 1 and 6 when the runner-up arrives there. n8 and n0 hold it down.' },
  13: { name: 'A faint helper for position 5', text: 'Adds a small vote for position 5 when the runner-up arrives there. It reads the record keeper and n8.' },
  9: { name: 'A faint echo of n5', text: 'Fires faintly for position 8, and n7 shuts it off whenever a number lands below the record. Its own vote changes nothing; what it feeds n1 and n6 still matters.' },
};
