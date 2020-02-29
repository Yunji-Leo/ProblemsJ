package actions;

import java.util.ArrayDeque;
import java.util.Deque;

public class BitTiger2 {
    class MinStack {
        Deque<Integer> stack;
        Deque<Integer> min;

        /**
         * initialize your data structure here.
         */
        public MinStack() {
            stack = new ArrayDeque<>();
            min = new ArrayDeque<>();
        }

        public void push(int x) {
            stack.push(x);
            if (!min.isEmpty() && min.peek() < x) {
                min.push(min.peek());
            } else {
                min.push(x);
            }
        }

        public void pop() {
            stack.pop();
            min.pop();
        }

        public int top() {
            return stack.peek();
        }

        public int getMin() {
            return min.peek();
        }
    }
}
