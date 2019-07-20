package problems;

import org.junit.runner.RunWith;

import java.util.List;

import static org.junit.Assert.*;

public class SolutionTest {
    @org.junit.Test
    public void testMethod() {
        Solution s = new Solution();
        s.threeSumClosest(new int[]{-1, 2, 1, -4}, 1);
        assertTrue(s.isPalindrome(121));
    }
}
